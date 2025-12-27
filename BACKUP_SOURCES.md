# Backup Data Sources - Konfigurationsguide

Appen använder nu flera metoder för att hämta aktiekurser för bättre tillförlitlighet.

## Nuvarande Datakällor

### 1. Yahoo Finance (Primär källa)
- **Ingen API-nyckel krävs**
- Använder flera metoder:
  - `info` dictionary (currentPrice, regularMarketPrice)
  - `fast_info` (snabbare, mindre data)
  - `history` (1 minut intervall)
  - `history` (5 dagar, senaste stängningskurs)

### 2. Alpha Vantage (Backup - Valfritt)
- **Gratis API-nyckel** från: https://www.alphavantage.co/support/#api-key
- 5 API-anrop per minut (gratis tier)
- 500 anrop per dag (gratis tier)

### 3. Finnhub (Backup - Valfritt)
- **Gratis API-nyckel** från: https://finnhub.io/
- 60 API-anrop per minut (gratis tier)
- Bra för internationella aktier

## Konfiguration

### För Lokal Körning

Skapa en fil `.streamlit/secrets.toml`:

```toml
# Alpha Vantage API Key (valfritt)
ALPHA_VANTAGE_API_KEY = "din_api_nyckel_här"

# Finnhub API Key (valfritt)
FINNHUB_API_KEY = "din_api_nyckel_här"
```

### För Streamlit Cloud

1. Gå till din app på Streamlit Cloud
2. Klicka på "Settings" (⚙️)
3. Klicka på "Secrets"
4. Lägg till:

```toml
ALPHA_VANTAGE_API_KEY = "din_api_nyckel_här"
FINNHUB_API_KEY = "din_api_nyckel_här"
```

## Hur Det Fungerar

1. **Första försöket**: Yahoo Finance med flera metoder
2. **Om Yahoo Finance misslyckas**: Försöker Alpha Vantage
3. **Om Alpha Vantage misslyckas**: Försöker Finnhub
4. **Om allt misslyckas**: Visar 0 eller "N/A"

## Fördelar

- ✅ **Högre tillförlitlighet**: Om en källa misslyckas, försöker nästa
- ✅ **Ingen API-nyckel krävs**: Fungerar med bara Yahoo Finance
- ✅ **Gratis backup**: Båda backup-källorna har gratis tier
- ✅ **Automatisk fallback**: Ingen manuell konfiguration behövs

## Tips

- **För svenska aktier**: Yahoo Finance fungerar oftast bra med `.ST` suffix
- **För amerikanska aktier**: Alla källor fungerar bra
- **Om kurser saknas**: Kontrollera att ticker-symbolen är korrekt
- **Rate limiting**: Gratis tier har begränsningar, men räcker för personlig användning

## Debugging

Om kurser fortfarande saknas:
1. Kontrollera ticker-symbolen (t.ex. `investor-b.st` för svenska)
2. Testa tickern direkt på Yahoo Finance
3. Kontrollera om API-nycklar är korrekt konfigurerade (om du använder backup-källor)

