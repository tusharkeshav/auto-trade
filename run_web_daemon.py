import time
from dashboard.web_server import BotState, start_web_server

if __name__ == "__main__":
    state = BotState()
    start_web_server(state, port=8502)
    print("AutoTrader Web Dashboard listening on 0.0.0.0:8502...")
    while True:
        time.sleep(60)
