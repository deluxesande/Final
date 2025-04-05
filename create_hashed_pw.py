import pickle
import streamlit_authenticator as stauth

# Define user credentials
names = ["Lucy"]
usernames = ["Lucy"]
passwords = ["password1"]  # Replace with your actual passwords

# Hash the passwords
hashed_passwords = stauth.Hasher(passwords).generate()

# Save the hashed passwords to a pickle file
file_path = "hashed_pw.pkl"
with open(file_path, "wb") as file:
    pickle.dump(hashed_passwords, file)

print(f"Hashed passwords saved to {file_path}")

