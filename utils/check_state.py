import retro

print("checking available states for mortal kombat ii...")
print()

try:
    states = retro.data.list_states('MortalKombatII-Genesis')
    print(f"found {len(states)} states:")
    for state in states:
        print(f"  - {state}")
except Exception as e:
    print(f"error: {e}")
    print("\ntrying to list all games...")
    games = retro.data.list_games()
    print(f"available games: {games}")
