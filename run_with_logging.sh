#!/bin/bash

# Record the command being executed
echo "$(date): Running command: $*" >> logs/session.log

# Execute the command and capture both stdout and stderr to session.log
"$@" 2>&1 | tee -a logs/session.log
