def load_users_and_passwords(filenames=["auth/users.txt", "auth/password.txt"]):
    users = []
    passwords = []

    with open(filenames[0], "r", encoding="utf-8") as file:
        users.extend([line.strip() for line in file.readlines()])

    with open(filenames[1], "r", encoding="utf-8") as file:
        passwords.extend([line.strip() for line in file.readlines()])

    return users, passwords

usernames, passwords = load_users_and_passwords()
names = usernames

credentials = {
    "usernames": {}
}

for key, username in enumerate(usernames):
    print(username)
    new_user = {username: {"name": names[key],"password": passwords[key]}}
    credentials["usernames"].update(new_user)

print(credentials)