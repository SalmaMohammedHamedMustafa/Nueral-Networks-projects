#!/bin/bash

# Configuration variables
USERNAME="salma"
SCRIPT_PATH="/home/salma/college/nueral_networks/Dr_mohsen/project/conductor.py"
VENV_PATH="/home/salma/college/nueral_networks/Dr_mohsen/project/venv"
SSH_AUTHORIZED_KEYS="/home/$USERNAME/.ssh/authorized_keys"
WORKING_DIR="/home/salma/college/nueral_networks/Dr_mohsen/project"
JSON_FILE="/home/salma/college/nueral_networks/Dr_mohsen/project/info.json"
LOG_FILE="/tmp/ssh_key_management.log"
REVOKE_SCRIPT="/home/$USERNAME/revoke_ssh_key.sh"

# Log function for easier debug
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo "$1"
}

log_message "Starting SSH key management script"

# Check if JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    log_message "Error: JSON file not found at $JSON_FILE"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    log_message "Error: jq is not installed. Install it with: sudo apt-get install jq"
    exit 1
fi

# Read the SSH public key from the JSON file
SSH_PUBLIC_KEY=$(jq -r '.ssh_public_key' "$JSON_FILE")

# Validate SSH public key
if [ -z "$SSH_PUBLIC_KEY" ] || [ "$SSH_PUBLIC_KEY" = "null" ]; then
    log_message "Error: 'ssh_public_key' field is missing or empty in $JSON_FILE"
    exit 1
fi

# Key is valid, now set up the revocation script
log_message "Creating revocation script at $REVOKE_SCRIPT"

cat > "$REVOKE_SCRIPT" << 'EOF'
#!/bin/bash

# This script removes the SSH key from authorized_keys file
# It's designed to be run by the SSH forced command

# Configuration
USERNAME="salma"
SSH_AUTHORIZED_KEYS="/home/$USERNAME/.ssh/authorized_keys"
LOG_FILE="/tmp/ssh_key_management.log"
JSON_FILE="/home/salma/college/nueral_networks/Dr_mohsen/project/info.json"
TEMP_FILE="/tmp/authorized_keys.tmp"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - REVOKE: $1" >> "$LOG_FILE"
}

log_message "Starting key revocation process"

# Check if the authorized_keys file exists
if [ ! -f "$SSH_AUTHORIZED_KEYS" ]; then
    log_message "Error: authorized_keys file not found at $SSH_AUTHORIZED_KEYS"
    exit 1
fi

# Get the key from JSON
if [ -f "$JSON_FILE" ]; then
    SSH_PUBLIC_KEY=$(jq -r '.ssh_public_key' "$JSON_FILE")
    
    if [ -n "$SSH_PUBLIC_KEY" ] && [ "$SSH_PUBLIC_KEY" != "null" ]; then
        # Extract the key fingerprint for more reliable matching
        TEMP_KEY_FILE="/tmp/temp_key_file_$$.pub"
        echo "$SSH_PUBLIC_KEY" > "$TEMP_KEY_FILE"
        KEY_FINGERPRINT=$(ssh-keygen -lf "$TEMP_KEY_FILE" 2>/dev/null | awk '{print $2}')
        rm -f "$TEMP_KEY_FILE"
        
        if [ -n "$KEY_FINGERPRINT" ]; then
            log_message "Key fingerprint to remove: $KEY_FINGERPRINT"
            
            # Strategy 1: Use simple pattern matching
            # Extract a unique part of the key for matching
            KEY_PART=$(echo "$SSH_PUBLIC_KEY" | awk '{print $2}' | cut -c1-20)
            BEFORE_COUNT=$(grep -c "$KEY_PART" "$SSH_AUTHORIZED_KEYS")
            log_message "Found $BEFORE_COUNT matches for key pattern in authorized_keys"
            
            if [ "$BEFORE_COUNT" -gt 0 ]; then
                # Make a backup first
                cp "$SSH_AUTHORIZED_KEYS" "$SSH_AUTHORIZED_KEYS.bak_$(date +%s)"
                
                # Remove lines containing the key pattern
                grep -v "$KEY_PART" "$SSH_AUTHORIZED_KEYS" > "$TEMP_FILE"
                cat "$TEMP_FILE" > "$SSH_AUTHORIZED_KEYS"
                rm -f "$TEMP_FILE"
                
                # Verify removal
                AFTER_COUNT=$(grep -c "$KEY_PART" "$SSH_AUTHORIZED_KEYS")
                log_message "After removal: $AFTER_COUNT matches remain"
                
                if [ "$AFTER_COUNT" -lt "$BEFORE_COUNT" ]; then
                    log_message "Key removal successful"
                else
                    log_message "Key removal failed - pattern approach"
                fi
            else
                log_message "Key pattern not found in authorized_keys"
            fi
        else
            log_message "Could not generate key fingerprint"
        fi
    else
        log_message "Invalid or empty SSH key in JSON file"
    fi
else
    log_message "JSON file not found"
fi

# Extra verification by checking if authorized_keys is empty
if [ ! -s "$SSH_AUTHORIZED_KEYS" ]; then
    log_message "WARNING: authorized_keys file is empty after operation"
fi

log_message "Key revocation process completed"
exit 0
EOF

# Make revocation script executable
chmod +x "$REVOKE_SCRIPT"
chown "$USERNAME:$USERNAME" "$REVOKE_SCRIPT"
log_message "Revocation script created and made executable"

# Extract a unique part of the key for the forced command
KEY_PART=$(echo "$SSH_PUBLIC_KEY" | awk '{print $2}' | cut -c1-20)

# Check if key already exists in authorized_keys
if grep -q "$KEY_PART" "$SSH_AUTHORIZED_KEYS"; then
    log_message "Key already exists in authorized_keys, removing it first"
    grep -v "$KEY_PART" "$SSH_AUTHORIZED_KEYS" > "/tmp/authorized_keys.tmp"
    cat "/tmp/authorized_keys.tmp" > "$SSH_AUTHORIZED_KEYS"
    rm -f "/tmp/authorized_keys.tmp"
fi

# Add the public key with script execution forced and automatic revocation
FORCED_COMMAND="command=\"cd $WORKING_DIR && source $VENV_PATH/bin/activate && python $SCRIPT_PATH && bash $REVOKE_SCRIPT\""
echo "$FORCED_COMMAND $SSH_PUBLIC_KEY" >> "$SSH_AUTHORIZED_KEYS"
log_message "Added SSH key with forced command and auto-revocation"

# Check if key was added successfully
if grep -q "$KEY_PART" "$SSH_AUTHORIZED_KEYS"; then
    log_message "SSH key successfully added to authorized_keys"
else
    log_message "ERROR: Failed to add SSH key to authorized_keys"
    exit 1
fi

# Set proper permissions on authorized_keys
chmod 600 "$SSH_AUTHORIZED_KEYS"
chown "$USERNAME:$USERNAME" "$SSH_AUTHORIZED_KEYS"
log_message "Set proper permissions on authorized_keys"

log_message "SSH key setup completed successfully"

# Instructions for manual testing
echo -e "\n===== Testing & Debugging Instructions ====="
echo "1. To manually revoke the SSH key, run: bash $REVOKE_SCRIPT"
echo "2. To check logs, run: cat $LOG_FILE"
echo "3. To see current authorized keys, run: cat $SSH_AUTHORIZED_KEYS"