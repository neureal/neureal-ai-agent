#!/usr/bin/env bash

[ $PRIV ] && pip3 install --user -e /outerdir/gym-trader
[ $PRIV ] && pip3 install --user -e /outerdir/neureal-ai-interfaces
[ $PRIV ] && pip3 install --user -e /outerdir/neureal-ai-util

#TEMPORARY
pip install mt5linux

[ $DEV ] && exec bash

if [ $PRIV ]
then
  /usr/bin/python /outerdir/neureal-ai-agent/agent_research.py
else
  /usr/bin/python /app/agent.py
fi
