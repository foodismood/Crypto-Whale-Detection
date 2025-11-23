# Crypto-Whale-Detection

## Kurzbeschreibung
Wir möchten die Preisveränderung von Kryptowährungen innerhalb von 15 Minuten nach einem Whale-Transfer prognostizieren.

## Problem Definition

In diesem Projekt untersuchen wir, wie stark der Markt auf große Whale-Transaktionen reagiert.  
Whales bewegen häufig mehrere Millionen US-Dollar und gelten daher als potenzielle Indikatoren für kurzfristige Preisbewegungen.  
Kryptomärkte reagieren oft empfindlich, wenn große Investoren („Whales“) Kapital zwischen Wallets und Börsen verschieben.

Unser Ziel ist es, die Preisänderung innerhalb von 15 Minuten nach einem Whale-Transfer vorherzusagen.  
Dafür kombinieren wir On-Chain-Blockchain-Daten mit Marktdaten von Binance und analysieren, ob Modelle kurzfristige Muster erkennen können, die auf Whale-Aktivitäten zurückzuführen sind.

## Projektziel

Wir möchten ein Modell entwickeln, das:

- kurzfristige Preisbewegungen erkennt, die durch den Einfluss großer On-Chain-Transaktionen entstehen,
- Whale-Aktivitäten mit Marktstruktur kombiniert,
- die prozentuale Kursänderung 15 Minuten nach einem Whale-Transfer so genau wie möglich prognostiziert.

## Input Data

### WhaleAlert On-Chain Daten

Für alle relevanten Transaktionen erfassen wir:

- `timestamp`
- `symbol` / Währung (z. B. BTC, ETH, USDT)
- `amount_usd` (Größe der Transaktion in USD)
- `amount_token` (Menge des Tokens)
- `from_type` (wallet, exchange)
- `to_type` (wallet, exchange)
- `transfer_direction` (Deposit / Withdrawal)
- `transaction_hash`

### Binance Market Data

Für jede Minute erfassen wir:

- `timestamp`
- `open`, `high`, `low`, `close`
- `volume`
- `vwap`

## Input Features

Wir führen die On-Chain-Signale und Marktdaten in einem gemeinsamen Feature-Set zusammen.

### Whale Features

- `transaction_usd_zscore`  
  → Wie extrem groß die Transaktion im Vergleich zu anderen Transaktionen ist  
- `transfer_direction_encoded`  
  → Codierung der Richtung (z. B. Deposit = 1, Withdrawal = 0)  
- wallet → exchange  
  → potenzieller Verkauf  
- exchange → wallet  
  → potenzieller Kauf  
- `whale_activity_last_10min`  
  → Anzahl großer Transfers im 10-Minuten-Fenster  
- `same_symbol_whale_volume_30min`  
  → kumuliertes Whale-Volumen der letzten 30 Minuten für dasselbe Symbol

### Price-Trend Features

- `return_1m`, `return_3m`, `return_5m`  
  → kurzfristiges Momentum  
- `price_slope_5m`  
  → lineare Steigung des Preises der letzten 5 Minuten  
- `EMA(20)`, `EMA(50)`  
  → Trendrichtung  
- `ema_distance` = `EMA20 - EMA50`  
  → Trendbeschleunigung

### Volatility Features

- `ATR(14)`  
  → Marktvolatilität  
- `realized_vol_10m`  
  → realisierte Volatilität der letzten 10 Minuten  
- `high_low_range`  
  → Kerzenvolatilität pro Minute (High–Low)

### Volume Features

- `volume_zscore_30m`  
  → Volume-Spike-Erkennung (Volumen im Vergleich zu den letzten 30 Minuten)  
- `vwap_deviation`  
  → Preisposition relativ zum volumengewichteten Durchschnittspreis (VWAP)  
- `volume_per_trade` *(falls verfügbar)*  

## Output

Für jede Whale-Transaktion erzeugen wir eine Zielgröße:

- Vorhersage der **prozentualen Preisänderung in den nächsten 15 Minuten**.
