# Phase 5b: profile and chat history

New: tables `user_profiles` and `chat_messages` (created automatically next to the Phase 4 tables,
so an existing `matbakhak.db` keeps working), `services/profile.py`, a `profile` agent, 3 profile tools.

- **Profile** (per user): mode (household / business), family size, favorite cuisines, disliked
  ingredients, diet, usual time limit, budget + period, business name and type, currency EGP.
- **Allergies stay in the existing memory file** (`recipe_agent.memory`), the same store the recipe
  search already reads. The profile agent saves them only when the user asks to save them, and the
  allergy words in the message are added by code even if the LLM missed them.
- **History**: each turn (message and answer) is saved per user. The last 6 turns are given to the
  router and the extractors only to resolve follow-ups ("and how much does it cost?"). The user can
  say "delete my chat history / profile / allergies" (the agent asks what exactly if unclear).
- The saved budget is used by the budget agent when the user does not give one.
- A database problem while loading or saving history never blocks an answer.

Not used yet: favorite cuisines, dislikes, diet, family size and the usual time limit are saved and
shown but the recipe search does not use them yet (only allergies are enforced).
Login is still not in the app: every browser shares one user id unless a username is provided.
