import open3d as o3d
import numpy as np
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import base64
import tempfile
import os
from getpass import getpass



def load_encrypted_pcd(encrypted_filename, cipher):
    """
    Decrypts an encrypted .ply file in memory and loads it as a point cloud.
    """
    with open(encrypted_filename, "rb") as file:
        encrypted_data = file.read()

    decrypted_data = cipher.decrypt(encrypted_data)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".ply") as temp_file:
        temp_file.write(decrypted_data)
        temp_file.flush()

        pcd = o3d.io.read_point_cloud(temp_file.name)

    return np.asarray(pcd.points), np.asarray(pcd.colors)


# Function to generate the key from a password
def generate_key_from_password(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def get_cypher():
    # Use the same fixed salt (DO NOT CHANGE)
    SALT = b'moqueca_e_camarao_salgadinha'
    
    # Option 1: Manually enter password
    password = os.environ.get('password', None)
    if not password:
        password = getpass('Enter decryption password: ')
    #password = input("Enter decryption password: ")
    
    # Option 2: Use environment variable (for automation)
    # password = os.getenv("PLY_DECRYPT_PASSWORD")
    
    key = generate_key_from_password(password, SALT)
    cypher = Fernet(key)
    return cypher
