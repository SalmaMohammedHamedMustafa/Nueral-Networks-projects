#!/bin/bash

# Configuration variables
USERNAME="salma"
SCRIPT_PATH="/home/salma/college/nueral_networks/Dr_mohsen/project/conductor2.py"
VENV_PATH="/home/salma/college/nueral_networks/Dr_mohsen/project/venv"
SSH_AUTHORIZED_KEYS="/home/$USERNAME/.ssh/authorized_keys"
TEMP_KEY_FILE="/tmp/user_key.pub"
API_KEY_FILE="/home/$USERNAME/.openai_api_key"


OPENAI_API_KEY="sk-your-actual-openai-api-key-here"  # Replace with your actual OpenAI API key

# Check if SSH server is installed and running
if ! command -v sshd &> /dev/null; then
    echo "Error: SSH server (sshd) is not installed. Install it with: sudo apt-get install openssh-server"
    exit 1
fi
if ! systemctl is-active --quiet ssh; then
    echo "Starting SSH server..."
    sudo systemctl start ssh
fi

# Ensure .ssh directory exists with correct permissions
mkdir -p "/home/$USERNAME/.ssh"
chmod 700 "/home/$USERNAME/.ssh"
chown "$USERNAME:$USERNAME" "/home/$USERNAME/.ssh"

# Ensure authorized_keys file exists with correct permissions
touch "$SSH_AUTHORIZED_KEYS"
chmod 600 "$SSH_AUTHORIZED_KEYS"
chown "$USERNAME:$USERNAME" "$SSH_AUTHORIZED_KEYS"

# Store OpenAI API key in a secure file
echo "export OPENAI_API_KEY='$OPENAI_API_KEY'" > "$API_KEY_FILE"
chmod 600 "$API_KEY_FILE"
chown "$USERNAME:$USERNAME" "$API_KEY_FILE"
echo "OpenAI API key stored securely in $API_KEY_FILE."

# Prompt for the participant's SSH public key
echo "Please paste the participant's SSH public key and press Enter (then Ctrl+D to finish):"
cat > "$TEMP_KEY_FILE"

if [ ! -s "$TEMP_KEY_FILE" ]; then
    echo "Error: No public key provided. Exiting."
    rm -f "$TEMP_KEY_FILE"
    exit 1
fi

# Add the public key with command restriction to run the interview script with API key
COMMAND="command=\"source $API_KEY_FILE && source $VENV_PATH/bin/activate && python $SCRIPT_PATH && exit\" $(cat $TEMP_KEY_FILE)"
echo "$COMMAND" >> "$SSH_AUTHORIZED_KEYS"
echo "Public key added with restriction to run $SCRIPT_PATH with OpenAI API key."

# Clean up temporary key file
rm -f "$TEMP_KEY_FILE"

# Get the machine's IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}' || curl -s ifconfig.me)
if [ -z "$IP_ADDRESS" ]; then
    echo "Error: Could not determine IP address. Please provide it manually."
    exit 1
fi

# Generate connection instructions
echo -e "\n=== Connection Instructions for the Participant ==="
echo "To start the interview:"
echo "1. Open a terminal on your computer."
echo "2. Use the following command to connect:"
echo "   ssh $USERNAME@$IP_ADDRESS"
echo "3. The interview script will run automatically."
echo "4. You will be disconnected when the interview is complete."
echo "5. Ensure your SSH private key is set up correctly."
echo -e "\nNote: Share these instructions securely with the participant."

# Reminder for revoking access
echo -e "\n=== To Revoke Access After the Interview ==="
echo "Remove the participant's key from $SSH_AUTHORIZED_KEYS:"
echo "1. Open the file: nano $SSH_AUTHORIZED_KEYS"
echo "2. Delete the line starting with 'command=\"source $API_KEY_FILE...'"
echo "3. Save and exit."
echo "Remove the API key file (optional, if no longer needed):"
echo "   rm $API_KEY_FILE"