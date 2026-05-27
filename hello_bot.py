"""Step 1 of TOURNAMENT.md — Hello bot (10 lines).

Open the browser tab first:
    https://ml.ferit.tech/?room=inva4

Then run:
    py hello_bot.py
"""
from game_client import RoomBot


def controller(obs):
    return 0.7, obs["navigation"]["heading_error"] * 0.5


bot = RoomBot("https://ml.ferit.tech", room="inva4", name="inva")
standings = bot.run(controller, hz=20.0)
print("Standings:", standings)
