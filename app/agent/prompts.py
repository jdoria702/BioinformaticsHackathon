import logging

logger = logging.getLogger(__name__)

# Simple tutor prompt (add more later):
SYSTEM_PROMPT = """
You are a bioinformatics tutor focused on one topic at a time.
Teach step by step.
Prefer short explanations first, then expand if needed.
After answering, ask one concept-check question.
If the student is confused, give an analogy.
Do not fabricate scientific details.
When a tool result is available, use it faithfully.
Return JSON with:
- answer
- follow_up_question
- key_points
- difficulty
"""

def build_tutor_prompt(topic: str, lesson_context: str, history, user_message: str) -> str:
    logger.info("Building tutor prompt...")

    # Check if we have history to retrieve:
    logger.debug("No chat history provided.")
    if not history:
        history_text = ""
    else:
        logger.debug(f"Formatting chat history with {len(history)} messages.")

        history_lines = []

        for message in history:
            try:
                history_lines.append(f"{message['role']}: {message['content']}")
            except KeyError as e:
                logger.warning(f"Malformed history message skipped: {message} (missing {e})")

        history_text = "\n".join(history_lines)

    logger.debug(
        f"Prompt components — topic='{topic}', "
        f"lesson_context_length={len(lesson_context)}, "
        f"user_message_length={len(user_message)}"
    )

    # Build the prompt:
    prompt = (
        f"You are a helpful bioinformatics tutor.\n"
        f"Topic: {topic}\n\n"
        f"Lesson context:\n{lesson_context}\n\n"
        f"Chat history:\n{history_text}\n\n"
        f"User message:\n{user_message}\n"
    )

    logger.debug(f"Final prompt length: {len(prompt)}")
    logger.debug(f"Full prompt:\n{prompt}")
    
    return prompt
