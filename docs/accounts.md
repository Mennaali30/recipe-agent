# Accounts: register and login

**Update: accounts now use email + password, with password reset and optional email confirmation. See docs/api.md (Login) for the current API. The text below describes the first version (username); old username accounts still log in.**

Each person has an account. Her profile, pantry, saved allergies and chat history belong to that account only.

- Username: 3-32 characters, a-z 0-9 _ . - (lower case). It is also the user id. No email is needed (no email is sent or stored), so there is no "forgot password" by email yet: to reset one, the owner removes the row from the `accounts` table and the person registers again, or uses "change password" while logged in.
- Password: 8-128 characters, stored as a salted scrypt hash. Never shown or logged.
- 5 wrong passwords for one username lock it for 5 minutes.
- An old user id that already has data (pantry, history, saved allergies) cannot be registered, so nobody can take over old data. The names `default`, `admin`, `root`, `system`, `me`, `null`, `none` are reserved.
- Token: sent as `Authorization: Bearer <token>`, valid 7 days, signed with `RECIPE_AUTH_SECRET` (set it, otherwise tokens stop working when the server restarts). There is no logout list: the app just forgets the token.

Endpoints: `POST /auth/register`, `POST /auth/login`, `GET /auth/me`, `POST /auth/password`.
Set `RECIPE_REQUIRE_LOGIN=1` so `/chat` and `/users/...` need the token and only work for the token's user (another user id gets 403). Without it the old behaviour stays (user id in the request), which is what the local app uses today.

Not done (future work): email or phone sign-up, password reset by email, logout that cancels a token, deleting an account with all its data, login screen in the UI.
