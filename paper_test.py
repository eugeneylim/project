"""
IBKR paper-trading smoke test using ib_async.

Prerequisites:
  - TWS or IB Gateway running in PAPER mode with the API enabled.
  - pip install ib_async  (inside your activated venv)

Connection notes (WSL -> Windows):
  - Set HOST to the Windows host IP that WSL sees:
      grep nameserver /etc/resolv.conf | awk '{print $2}'
  - Add that IP to TWS "Trusted IPs" and untick "localhost only".
  - PORT: paper TWS = 7497, paper Gateway = 4002.
"""

from ib_async import IB, Stock, MarketOrder, util

HOST = "10.255.255.254"  # <-- replace with your Windows host IP from resolv.conf
PORT = 7497          # 7497 = paper TWS, 4002 = paper Gateway
CLIENT_ID = 1        # any unique integer


def main():
    ib = IB()
    print(f"Connecting to {HOST}:{PORT} ...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print("Connected:", ib.isConnected())

    # --- 1. Account sanity check -------------------------------------------
    summary = ib.accountSummary()
    net_liq = next((r.value for r in summary if r.tag == "NetLiquidation"), "n/a")
    print(f"Net liquidation value: {net_liq}")

    # --- 2. Define a contract and pull a quote -----------------------------
    contract = Stock("AAPL", "SMART", "USD")
    ib.qualifyContracts(contract)

    ticker = ib.reqMktData(contract)
    ib.sleep(2)  # let a tick arrive
    price = ticker.marketPrice()
    print(f"AAPL market price: {price}")

    # --- 3. Place a small paper MARKET order -------------------------------
    order = MarketOrder("BUY", 1)
    trade = ib.placeOrder(contract, order)
    print("Order submitted, waiting for fill/status ...")

    # Wait for the order to reach a terminal-ish state
    for _ in range(20):
        ib.sleep(0.5)
        print("  status:", trade.orderStatus.status)
        if trade.orderStatus.status in ("Filled", "Cancelled", "Inactive"):
            break

    print("Final status:", trade.orderStatus.status)
    print("Filled:", trade.orderStatus.filled, "@ avg", trade.orderStatus.avgFillPrice)

    # --- 4. Show open positions --------------------------------------------
    ib.sleep(1)
    positions = ib.positions()
    print("\nPositions:")
    for p in positions:
        print(f"  {p.contract.symbol}: {p.position} @ {p.avgCost}")

    ib.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    util.logToConsole()  # comment out for quieter runs
    main()
