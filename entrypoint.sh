#!/usr/bin/env bash

[ $PRIV ] && pip3 install --user -e /outerdir/gym-trader
[ $PRIV ] && pip3 install --user -e /outerdir/neureal-ai-interfaces
[ $PRIV ] && pip3 install --user -e /outerdir/neureal-ai-util

if [ $PRIV ]
then
  cd /outerdir/neureal-ai-agent
  /usr/local/bin/tensorboard --bind_all --port 6006 --logdir /outerdir/neureal-ai-agent/logs serve &
else
  cd /app
  /usr/local/bin/tensorboard --bind_all --port 6006 --logdir /app/logs serve &
fi

[ $DEV ] && exec bash

if [ $PRIV ]
then
  /usr/bin/python /outerdir/neureal-ai-agent/agent_research.py
else
  /usr/bin/python /app/agent.py
fi
