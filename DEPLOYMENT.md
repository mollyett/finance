# Deployment Guide - Streamlit Portfolio Tracker

## 🚀 Lokal Körning (Local Development)

### Steg 1: Installera Dependencies

Öppna terminalen i projektmappen och kör:

```bash
cd /Users/mattiasgustafsson/Documents/Programmering/Finance/finance
pip install -r requirements.txt
```

Eller om du använder pip3:

```bash
pip3 install -r requirements.txt
```

### Steg 2: Kör Appen

```bash
streamlit run app.py
```

Appen öppnas automatiskt i din webbläsare på `http://localhost:8501`

Om det inte öppnas automatiskt, öppna manuellt:
- Gå till: http://localhost:8501

### Steg 3: Testa Appen

1. Klicka på "➕ Add Transaction" i sidebar
2. Lägg till en test-transaktion (t.ex. AAPL)
3. Gå till "📊 Overview" för att se din portfölj

---

## ☁️ Deployment till Streamlit Cloud

### Steg 1: Skapa GitHub Repository

1. Gå till [GitHub](https://github.com) och skapa ett nytt repository
2. Döp det t.ex. till `portfolio-tracker`
3. **Viktigt**: Välj **Public** (Streamlit Cloud kräver public repos för free tier)

### Steg 2: Pusha Koden till GitHub

I terminalen, från projektmappen:

```bash
# Initiera git (om inte redan gjort)
git init

# Lägg till alla filer
git add .

# Gör första commit
git commit -m "Initial commit - Portfolio Tracker"

# Lägg till remote repository (ersätt med ditt repo-URL)
git remote add origin https://github.com/DITT-ANVÄNDARNAMN/portfolio-tracker.git

# Pusha till GitHub
git branch -M main
git push -u origin main
```

### Steg 3: Deploya till Streamlit Cloud

1. Gå till [Streamlit Cloud](https://streamlit.io/cloud)
2. Logga in med ditt GitHub-konto
3. Klicka på **"New app"**
4. Fyll i:
   - **Repository**: Välj ditt repository (`portfolio-tracker`)
   - **Branch**: `main` (eller `master`)
   - **Main file path**: `app.py`
5. Klicka på **"Deploy"**

### Steg 4: Vänta på Deployment

Streamlit Cloud kommer automatiskt:
- Installera alla dependencies från `requirements.txt`
- Köra appen
- Ge dig en URL (t.ex. `https://portfolio-tracker.streamlit.app`)

---

## ⚠️ Viktiga Noteringar för Streamlit Cloud

### Database Storage

**Viktigt**: SQLite-databasen sparas lokalt i `.data/` mappen. På Streamlit Cloud:

- **Free tier**: Databasen raderas när appen går inaktiv (efter 7 dagar inaktivitet)
- **Team tier**: Data persisterar bättre

**Alternativ för Production**:
- Använd en extern databas (PostgreSQL, MySQL) via Streamlit Secrets
- Eller använd CSV-export/import för backup

### Secrets Configuration (Valfritt)

Om du vill använda extern databas, skapa `.streamlit/secrets.toml`:

```toml
[postgres]
host = "your-host"
port = 5432
database = "your-db"
username = "your-user"
password = "your-password"
```

Lägg sedan till secrets i Streamlit Cloud dashboard.

---

## 🔧 Felsökning

### Problem: "ModuleNotFoundError"

**Lösning**: Se till att alla dependencies är i `requirements.txt`:

```bash
pip install streamlit yfinance pandas plotly numpy
pip freeze > requirements.txt
```

### Problem: "Database locked"

**Lösning**: Detta kan hända om flera instanser körs. Stäng alla Streamlit-instanser och starta om.

### Problem: "Ticker not found"

**Lösning**: 
- Kontrollera att ticker-symbolen är korrekt
- För svenska aktier, använd `.ST` suffix (t.ex. `investor-b.st`)
- Vissa tickers kan sakna data i yfinance

### Problem: Appen laddar långsamt

**Lösning**: 
- Data cachar automatiskt (5 min för stocks, 1 timme för currency)
- Första laddningen kan ta längre tid
- Överväg att begränsa antal tickers om det är många

---

## 📝 Checklista innan Deployment

- [ ] Alla filer är committade till git
- [ ] `requirements.txt` innehåller alla dependencies
- [ ] `.streamlit/config.toml` finns
- [ ] `.gitignore` inkluderar `.data/` och `*.db` (för att inte committa databasen)
- [ ] Testat appen lokalt
- [ ] Repository är public (för free tier)

---

## 🎯 Quick Start Commands

```bash
# Installera dependencies
pip install -r requirements.txt

# Kör lokalt
streamlit run app.py

# Eller med specifik port
streamlit run app.py --server.port 8502
```

---

## 💡 Tips

1. **Lokal utveckling**: Använd `streamlit run app.py` för snabb iteration
2. **Hot reload**: Streamlit laddar om automatiskt när du sparar filer
3. **Debugging**: Använd `st.write()` eller `st.sidebar.write()` för att debugga
4. **Performance**: För många tickers, överväg att öka cache-tiden i `data_fetcher.py`

---

## 📞 Support

Om du stöter på problem:
1. Kontrollera Streamlit Cloud logs i dashboard
2. Testa lokalt först för att isolera problemet
3. Kontrollera att alla dependencies är korrekta




