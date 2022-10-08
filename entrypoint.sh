#!/usr/bin/env bash

#pip3 install --user gym-trader@git+ssh://git@github.com/wilbown/gym-trader.git
#pip3 install --user neureal-ai-interfaces@git+ssh://git@github.com/wilbown/neureal-ai-interfaces.git
[ $PRIV ] && pip3 install --user -e /outerdir/gym-trader
[ $PRIV ] && pip3 install --user -e /outerdir/neureal-ai-interfaces

[ $DEV ] && exec bash

#TEMPORARY
pip install mt5linux

if [ $PRIV ]
then
  /usr/bin/python /outerdir/neureal-ai-agent/agent_research.py
else
  /usr/bin/python /app/agent.py
fi
