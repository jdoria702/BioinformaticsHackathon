import logging

from app.services.lesson_service import LessonService
from app.services.llm_service import LLMService
from app.services.session_service import SessionService
from app.agent.prompts import build_tutor_prompt

logger = logging.getLogger(__name__)

class BioTutorAgent:

    def __init__(self):
        self.llm_service = LLMService()
        self.lesson_service = LessonService()
        self.session_service = SessionService()

    def respond(self, user_message: str, session_id: str, topic:str) -> dict:
        logger.debug("BioTutorAgent.respond called (session_id=%s, topic=%s)", session_id, topic)

        # Our agent should store user sessions (each new chat has its own session):
        logger.debug("Fetching history...")
        history = self.session_service.get_history(session_id) 

        # Our agent should also remember what the lesson the user requested was about:
        logger.debug("Fetching lesson context...")
        lesson_context = self.lesson_service.get_topic_context(topic)

        # Create the prompt that is passed to the agent:
        logger.debug("Building prompt...")
        prompt = build_tutor_prompt(
            topic=topic,
            lesson_context=lesson_context,
            history=history,
            user_message=user_message
        )

        # Use the LLM of the agent to generate an appropriate response:
        logger.debug("Generating LLM response...")
        llm_result = self.llm_service.generate(prompt)

        # Update the chat session:
        logger.debug("Appending messages to session...")
        self.session_service.append(session_id, "user", user_message)
        self.session_service.append(session_id, "assistant", llm_result["answer"])

        return llm_result
