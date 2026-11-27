# Approval for sensitive actions

Some actions cannot be undone, so the assistant asks first. It does not run them.

- Needs approval by default: `pantry_discard` (throw away all of an item), `pantry_adjust` (set an exact amount), `profile_clear` (delete allergies, profile or chat history).
- Flow: the tool call is saved as ONE pending action per user (table `pending_actions`, expires after 10 minutes). The answer shows what will change and asks for "تأكيد" / "yes" or "لا" / "no".
- The next message decides: a clear yes runs exactly the saved action, a clear no cancels, any other message drops it (nothing runs). Only the same user can confirm. Words are matched on the whole short message ("نعم بس زودي" is not a yes).
- Fails closed: no database, or any error while saving, means the action does not run.
- Change the list: set `RECIPE_APPROVAL_TOOLS="pantry_discard,pantry_adjust,profile_clear,pantry_add"` (comma list, replaces the default) or `none` to turn it off. `pantry_add` (recording a purchase) is off by default.
