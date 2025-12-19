

import json
import time
import os
import csv
import signal
import sys
from datetime import datetime, timedelta
from websocket import create_connection

# Force UTF-8 encoding for stdout/stderr
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Try dxFeed demo endpoint
DXFEED_URL = "wss://demo.dxfeed.com/dxlink-ws"

# Global shutdown flag
shutdown_requested = False

# Test output immediately
print("=" * 60)
print("SPXW Time and Sales Collector - Starting")
print("=" * 60)
sys.stdout.flush()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    print(f"\n[SIGNAL] Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


def format_trade_time(epoch_ms):
    """Returns tuple of (date, time) with milliseconds"""
    if not epoch_ms:
        return "N/A", "N/A"
    dt = datetime.fromtimestamp(epoch_ms / 1000)
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
    return date_str, time_str


def format_timeand_sale_flags(flags):
    """Convert TimeAndSale eventFlags to human-readable format"""
    if flags is None:
        return "N/A"
    
    flag_descriptions = []
    
    if flags & 1:
        flag_descriptions.append("ETH")
    if flags & 2:
        flag_descriptions.append("SPREAD")
    if flags & 4:
        flag_descriptions.append("ISO")
    if flags & 8:
        flag_descriptions.append("OTC")
    if flags & 16:
        flag_descriptions.append("SINGLE")
    if flags & 32:
        flag_descriptions.append("COMPOSITE")
    if flags & 64:
        flag_descriptions.append("CONDITIONAL")
    
    return "|".join(flag_descriptions) if flag_descriptions else "REGULAR"


def parse_json_double(value):
    """Parse JSONDouble values that could be numbers or special strings"""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        if value in ["NaN", "Infinity", "-Infinity"]:
            return value
        try:
            return float(value)
        except ValueError:
            return value
    return "N/A"


def get_todays_expiration():
    """Generate today's expiration date in YYMMDD format"""
    return datetime.now().strftime("%y%m%d")


def generate_spxw_options(center_price, otm_percentage=0.15, increment=5, expiration=None):
    """Generate SPXW options symbols based on percentage OTM
    
    Args:
        center_price: Current SPX price
        otm_percentage: How far OTM to go (0.15 = 15%)
        increment: Strike spacing (5 for SPX)
        expiration: YYMMDD format, defaults to today
    """
    if expiration is None:
        expiration = get_todays_expiration()
    
    # Calculate OTM distance in points
    otm_points = center_price * otm_percentage
    
    # Calculate strike range
    start = center_price - otm_points
    end = center_price + otm_points
    
    # Round to increment
    start = (start // increment) * increment
    end = ((end // increment) + 1) * increment

    strikes = []
    current_strike = start
    while current_strike <= end:
        strikes.append(int(current_strike))
        current_strike += increment

    options = []
    for strike in strikes:
        options.append(f".SPXW{expiration}C{strike}")
        options.append(f".SPXW{expiration}P{strike}")

    print(f"\n[STRIKES] Strike Range Calculation:")
    print(f"          SPX Price: {center_price}")
    print(f"          OTM %: {otm_percentage*100}%")
    print(f"          OTM Points: +/-{otm_points:.0f}")
    print(f"          Strike Range: {int(start)} to {int(end)}")
    print(f"          Number of Strikes: {len(strikes)}")
    print(f"          Total Symbols (Calls + Puts): {len(options)}")

    return options


# Track last successful save time globally
last_successful_save = time.time()

def save_data_to_csv(time_and_sales_buffer, spx_quotes_buffer):
    """Save TimeAndSale events and SPX quotes to CSV file."""
    global last_successful_save

    if not time_and_sales_buffer and not spx_quotes_buffer:
        return False

    # Ensure the historical data directory exists
    os.makedirs("historical data", exist_ok=True)

    filename = os.path.join("historical data", f"spxw_tas_data_{datetime.now().strftime('%Y-%m-%d')}.csv")

    try:
        file_exists = os.path.exists(filename)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "date",
                    "time",
                    "symbol",
                    "event_type",
                    "price",
                    "size",
                    "exchange",
                    "bid_price",
                    "ask_price",
                    "time_and_sale_flags",
                    "sequence",
                    "exchange_sale_conditions",
                    "aggressor_side",
                    "spx_price",
                    "spx_bid",
                    "spx_ask",
                ],
            )
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writeheader()
            
            # Write TimeAndSale events
            if time_and_sales_buffer:
                writer.writerows(time_and_sales_buffer)
                print(f"[SAVE] Saved {len(time_and_sales_buffer)} TAS events -> {filename}")
                time_and_sales_buffer.clear()
            
            # Write SPX quotes
            if spx_quotes_buffer:
                writer.writerows(spx_quotes_buffer)
                print(f"[SAVE] Saved {len(spx_quotes_buffer)} SPX quotes -> {filename}")
                spx_quotes_buffer.clear()

        last_successful_save = time.time()
        return True

    except PermissionError:
        minutes_since_save = (time.time() - last_successful_save) / 60
        total_items = len(time_and_sales_buffer) + len(spx_quotes_buffer)
        print(f"[WARN] File in use ({filename}) -- keeping data in memory for next save.")
        print(f"[MEMORY] {total_items} items in memory, "
              f"{minutes_since_save:.1f} min since last successful save.")
        return False

    except Exception as e:
        print(f"[ERROR] Error writing {filename}: {e}")
        return False


def main():
    global shutdown_requested

    print("[MAIN] Entering main function...")
    sys.stdout.flush()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[CONNECT] Connecting to dxFeed demo endpoint: {DXFEED_URL}")
    print("[INFO] Note: Demo endpoint provides 15-minute delayed data")
    sys.stdout.flush()
    try:
        ws = create_connection(DXFEED_URL, timeout=30)
        print("[OK] Connected to dxFeed Demo API...")
    except Exception as e:
        print(f"[ERROR] Failed to connect to demo endpoint: {e}")
        return

    print("[SETUP] Setting up connection...")
    
    ws.send(json.dumps({
        "type": "SETUP",
        "channel": 0,
        "keepaliveTimeout": 60,
        "acceptKeepaliveTimeout": 60,
        "version": "1.0.0"
    }))
    print("-> Sent SETUP")

    try:
        setup_response = json.loads(ws.recv())
        print(f"[SETUP] Setup response: {setup_response}")
    except Exception as e:
        print(f"[WARN] Setup response error: {e}")
        print("[INFO] Continuing anyway...")

    # FEED CHANNEL
    ws.send(json.dumps({
        "type": "CHANNEL_REQUEST",
        "channel": 1,
        "service": "FEED",
        "parameters": {"contract": "AUTO"}
    }))
    print("-> Sent CHANNEL_REQUEST")

    try:
        channel_response = json.loads(ws.recv())
        print(f"[SETUP] Channel response: {channel_response}")
        if channel_response.get("type") == "CHANNEL_OPENED" and channel_response.get("channel") == 1:
            print("[OK] Channel opened.")
    except Exception as e:
        print(f"[WARN] Channel response error: {e}")
        print("[INFO] Continuing anyway...")

    # Get SPX midpoint
    print("\n[DATA] Fetching SPX midpoint from demo feed...")
    ws.send(json.dumps({
        "type": "FEED_SUBSCRIPTION",
        "channel": 1,
        "add": [{"symbol": "SPX", "type": "Quote"}]
    }))

    spx_price = None
    for _ in range(10):
        try:
            msg = json.loads(ws.recv())
            print(f"[DEBUG] Demo message: {msg}")
            if msg.get("type") == "FEED_DATA":
                for data in msg.get("data", []):
                    if data.get("eventSymbol") == "SPX" and data.get("eventType") == "Quote":
                        bid = data.get("bidPrice")
                        ask = data.get("askPrice")
                        if bid and ask:
                            spx_price = round(((bid + ask) / 2) / 5) * 5
                            print(f"[OK] SPX midpoint = {spx_price}")
                            break
            if spx_price:
                break
        except Exception as e:
            print(f"[WARN] Error receiving demo data: {e}")
            break

    if not spx_price:
        spx_price = 6800
        print("[WARN] Using default SPX price: 6800")

    # Generate SPXW options with today's expiration - 15% OTM each direction
    expiration = get_todays_expiration()
    print(f"[DATE] Using today's expiration: {expiration}")
    options = generate_spxw_options(spx_price, otm_percentage=0.15, increment=5, expiration=expiration)
    print(f"[INFO] Sample symbols: {options[:4]}")
    print(f"       Lowest strike: {options[0]}")
    print(f"       Highest strike: {options[-1]}")

    # Subscribe to TimeAndSale ONLY + SPX Quote
    print("\n[SUBSCRIBE] Subscribing to Time and Sales events...")
    
    tas_events = []
    for sym in options:
        tas_events.append({"symbol": sym, "type": "TimeAndSale"})
    
    # Add SPX Quote subscription
    tas_events.append({"symbol": "SPX", "type": "Quote"})
    
    ws.send(json.dumps({
        "type": "FEED_SUBSCRIPTION",
        "channel": 1,
        "add": tas_events
    }))
    print(f"[OK] Subscribed to {len(tas_events)} events (TAS + SPX Quote)")
    print("[INFO] Monitoring Time and Sales feed...")

    # Main loop - streamlined for TAS only
    latest_spx_quote = {}
    time_and_sales_buffer = []
    spx_quotes_buffer = []
    last_save_time = time.time()
    last_spx_quote_time = time.time()
    
    # Statistics
    stats = {
        'TimeAndSale': 0,
        'Quote': 0,
        'other': 0
    }
    
    last_stats_time = time.time()
    tas_print_counter = 0
    
    print("\n[START] Starting Time and Sales collection...")
    print("        Send SIGTERM to stop gracefully\n")

    try:
        while not shutdown_requested:
            try:
                msg = json.loads(ws.recv())
                msg_type = msg.get("type")

                if msg_type == "FEED_DATA":
                    for data in msg.get("data", []):
                        etype = data.get("eventType")
                        sym = data.get("eventSymbol", "")
                        
                        # Update statistics
                        if etype == "TimeAndSale":
                            stats['TimeAndSale'] += 1
                        elif etype == "Quote" and sym == "SPX":
                            stats['Quote'] += 1
                        else:
                            stats['other'] += 1

                        # Process TimeAndSale events
                        if etype == "TimeAndSale":
                            date_str, time_str = format_trade_time(data.get("time"))
                            
                            time_and_sales_buffer.append({
                                "date": date_str,
                                "time": time_str,
                                "symbol": sym,
                                "event_type": "TIME_AND_SALE",
                                "price": parse_json_double(data.get("price")),
                                "size": parse_json_double(data.get("size")),
                                "exchange": data.get("exchangeCode", "N/A"),
                                "bid_price": parse_json_double(data.get("bidPrice")),
                                "ask_price": parse_json_double(data.get("askPrice")),
                                "time_and_sale_flags": format_timeand_sale_flags(data.get("eventFlags")),
                                "sequence": data.get("sequence", "N/A"),
                                "exchange_sale_conditions": data.get("exchangeSaleConditions", "N/A"),
                                "aggressor_side": data.get("aggressorSide", "N/A"),
                                "spx_price": latest_spx_quote.get("midpoint", "N/A"),
                                "spx_bid": latest_spx_quote.get("bid", "N/A"),
                                "spx_ask": latest_spx_quote.get("ask", "N/A"),
                            })
                            
                            tas_print_counter += 1
                            if tas_print_counter % 100 == 0:
                                print(f"[DATA] TAS: {tas_print_counter} events collected, Buffer: {len(time_and_sales_buffer)}")

                        # Process SPX Quote
                        elif etype == "Quote" and sym == "SPX":
                            bid = data.get("bidPrice")
                            ask = data.get("askPrice")
                            quote_time = data.get("time")

                            if bid and ask:
                                latest_spx_quote = {
                                    "bid": bid,
                                    "ask": ask,
                                    "midpoint": (bid + ask) / 2,
                                    "time": quote_time
                                }

                elif msg_type == "KEEPALIVE":
                    ws.send(json.dumps({"type": "KEEPALIVE", "channel": 0}))

            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"[ERROR] General error: {e}")
                continue

            # Print statistics every 30 seconds
            if time.time() - last_stats_time >= 30:
                print(f"\n[STATS] FEED STATISTICS (last 30s):")
                print(f"        TimeAndSale: {stats['TimeAndSale']:,}")
                print(f"        SPX Quote: {stats['Quote']:,}")
                print(f"        Other: {stats['other']:,}")
                print(f"        TAS Buffer: {len(time_and_sales_buffer):,} events")
                print(f"        SPX: {latest_spx_quote.get('midpoint', 'N/A')}")
                
                stats = {k: 0 for k in stats}
                last_stats_time = time.time()

            # Save SPX quote every minute
            if time.time() - last_spx_quote_time >= 60:
                if latest_spx_quote:
                    # Data is 15 minutes delayed, so use current time minus 15 minutes
                    adjusted_time = datetime.now() - timedelta(minutes=15)
                    date_str = adjusted_time.strftime("%Y-%m-%d")
                    time_str = adjusted_time.strftime("%H:%M:%S.%f")[:-3]

                    spx_quotes_buffer.append({
                        "date": date_str,
                        "time": time_str,
                        "symbol": "SPX",
                        "event_type": "SPX_QUOTE",
                        "price": latest_spx_quote.get("midpoint", "N/A"),
                        "size": "N/A",
                        "exchange": "QUOTE",
                        "bid_price": latest_spx_quote.get("bid", "N/A"),
                        "ask_price": latest_spx_quote.get("ask", "N/A"),
                        "time_and_sale_flags": "N/A",
                        "sequence": "N/A",
                        "exchange_sale_conditions": "N/A",
                        "aggressor_side": "N/A",
                        "spx_price": latest_spx_quote.get("midpoint", "N/A"),
                        "spx_bid": latest_spx_quote.get("bid", "N/A"),
                        "spx_ask": latest_spx_quote.get("ask", "N/A"),
                    })
                last_spx_quote_time = time.time()

            # Save all data every 60s
            if time.time() - last_save_time >= 60:
                save_data_to_csv(time_and_sales_buffer, spx_quotes_buffer)
                last_save_time = time.time()

    finally:
        # Cleanup when shutdown is requested or error occurs
        print("\n[STOP] Stopping Time and Sales collection...")
        print("[STATS] FINAL STATISTICS:")
        for event_type, count in stats.items():
            print(f"        {event_type}: {count:,}")
        save_data_to_csv(time_and_sales_buffer, spx_quotes_buffer)
        try:
            ws.close()
        except:
            pass
        print("[EXIT] Goodbye!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPXW Time and Sales Collector")
    print("Python script starting...")
    print("=" * 60 + "\n")
    sys.stdout.flush()
    main()
