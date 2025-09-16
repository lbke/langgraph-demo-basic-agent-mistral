"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
# in older Python "Dict" was also imported to use generics, now dict supports them
from typing import Any

from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph, END
from langgraph.types import interrupt


# Pas utilisé dans cet application
# class Context(TypedDict):
#     """Context parameters for the agent.
#
#     Set these when creating assistants OR when invoking the graph.
#     See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
#     """
#     my_configurable_param: str

model_name = "mistral-small-latest"
moderation_model_name = "mistral-small-latest"

voyages_possibles = [
    {
        "nom": "Randonnée camping en Lozère",
        "labels": ["sport", "montagne", "campagne"],
        "accessibleHandicap": "non"
    },
    {
        "nom": "5 étoiles à Chamonix option fondue",
        "labels": ["montagne", "détente"],
        "accessibleHandicap": "oui"
    },
    {
        "nom": "5 étoiles à Chamonix option ski",
        "labels": ["montagne", "sport"],
        "accessibleHandicap": "non"
    },
    {
        "nom": "Palavas de paillotes en paillotes",
        "labels": ["plage", "ville", "détente", "paillote"],
        "accessibleHandicap": "oui"
    },
    {
        "nom": "5 étoiles en rase campagne",
        "labels": ["campagne", "détente"],
        "accessibleHandicap": "oui"
    }
]


@dataclass
class Criteres:
    # /!\ We can't initialize mutable objects directly in Python,
    # hence a lambda
    # We could also use "post_init"
    # Or init the dict directly in the agent code when needed
    criteres: dict[str, bool | None] = field(default_factory=lambda: {
        "plage": None,
        "montagne": None,
        "ville": None,
        "sport": None,
        "detente": None,
        "acces_handicap": None,
    })

    def au_moins_un_critere_rempli(self) -> bool:
        return any(v is not None for v in self.criteres.values())


@dataclass
class State(Criteres):
    """
    Etat de l'agent
    """
    message_utilisateur: str = ""
    message_ia: str = "",
    injection: bool = False
    erreur_ia: bool = False
    done: bool = False


# TRAITEMENT DES CRITERES

async def bonjour(state: State) -> dict[str, Any]:
    reponse = interrupt({
        "message_ia": "Pour mieux vous aider, pourriez-vous me dire quels sont vos critères de voyage? Par exemple, préférez-vous la plage, la montagne, la ville, le sport, la détente, ou avez-vous besoin d'un accès handicapé?"
    })
    return {
        "message_utilisateur": reponse
    }


async def mise_a_jour_criteres(state: State) -> dict[str, Any]:
    # Met à jour les critères
    prompt_criteres = f"""
    L'utilisateur vient d'envoyer le message suivant:
    <message_utilisateur>
    {state.message_utilisateur}
    </message_utilisateur>
    Les critères actuels sont:
    <criteres>
    {state.criteres}
    </criteres>
    Met à jour le tableau des critères avec True, False ou None selon le cas.

    Critères possibles: {state.criteres.keys()}
    """
    model = init_chat_model(model=model_name, model_provider="mistralai")
    model_with_output = model.with_structured_output(Criteres)
    criteres_a_jour = await model_with_output.ainvoke(prompt_criteres)
    return {
        "criteres": criteres_a_jour["criteres"],
    }


# CONVERSATION AVEC L'UTILISATEUR

async def choix_voyage_et_question(state: State) -> dict[str, Any]:
    # Si une injection a été détectée, on passe directement à la suite
    if state.injection:
        return state
    # Si une hallucination a été détectée, on passe à la suite
    if state.erreur_ia:
        return state
    # NOTE: on pourrait utiliser une "Command" pour sauter vers un noeud d'erreur directement

    model = init_chat_model(model=model_name, model_provider="mistralai")
    prompt_reponse = f"""
    Tu es un agent de voyage.
    Adresse toi à l'utilisateur de façon polie et professionnelle.

    L'utilisateur vient d'envoyer le message suivant:
    <message_utilisateur >
    {state.message_utilisateur}
    </message_utilisateur >
    Fait référence à ce message dans ta réponse en une phrase. Si le message de l'utilisateur n'a pas de sens, indique que tu n'as pas compris sa demande, mais que tu vas t'efforcer de trouver le voyage idéal tout de même.
    """

    prompt_reponse += f"""
        Les critères suivants ont été déduits pour son voyage idéal:
        <criteres >
        {state.criteres}
        </criteres >

        Parmi les critères possibles suivants:
        <critères_possibles >
        {state.criteres.keys()}
        </critères_possibles >

        Propose-lui le voyage idéal parmis les voyages disponibles et en respectant ses critères:
        <criteres >
        {state.criteres}
        </criteres >
        <voyages_possibles >
        {voyages_possibles}
        </voyages_possibles >
        """
    res = await model.ainvoke(prompt_reponse)
    return {
        "message_ia": res.content,
    }


# NOTE: le graphe est rejoué entièrement à chaque message utilisateur
# mais son state est préservé
# Voir la version "interrupt" pour une alternative en un seul tour
graph = (
    StateGraph(State)
    .add_node(bonjour)  # facultatif, pour initier la conversation
    .add_node(mise_a_jour_criteres)
    .add_node(choix_voyage_et_question)
    .add_edge(START, bonjour.__name__)
    .add_edge(bonjour.__name__, mise_a_jour_criteres.__name__)
    .add_edge(mise_a_jour_criteres.__name__, choix_voyage_et_question.__name__)
    .add_edge(choix_voyage_et_question.__name__, END)
    .compile(name="Agent de voyage (avec interruption)")
)
