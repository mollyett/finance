# 🚀 Snabbstart - Portfolio Tracker

## Kör Appen Lokalt (3 steg)

### 1. Installera Dependencies (om inte redan gjort)

```bash
pip install -r requirements.txt
```

### 2. Starta Appen

```bash
streamlit run app.py
```

### 3. Öppna i Webbläsare

Appen öppnas automatiskt på: **http://localhost:8501**

Om den inte öppnas automatiskt, kopiera URL:en från terminalen.

---

## ✅ Testa Appen

1. **Lägg till din första transaktion:**
   - Klicka på "➕ Add Transaction" i sidebar
   - Fyll i:
     - Ticker: `AAPL` (eller `investor-b.st` för svenska aktier)
     - Purchase Date: Välj datum
     - Purchase Price: `150.00`
     - Quantity: `10`
     - Currency: `USD` (eller `SEK`)
   - Klicka "Add Transaction"

2. **Se din portfölj:**
   - Gå till "📊 Overview"
   - Se metrics, tabeller och visualiseringar

3. **Sätt target allocations:**
   - Gå till "⚙️ Settings"
   - Sätt target % för varje innehav
   - Gå tillbaka till Overview för rebalance-förslag

---

## ☁️ Deploya till Streamlit Cloud

### Steg 1: Pusha till GitHub

```bash
git init
git add .
git commit -m "Portfolio Tracker app"
git remote add origin https://github.com/DITT-ANVÄNDARNAMN/repo-namn.git
git push -u origin main
```

### Steg 2: Deploya

1. Gå till [streamlit.io/cloud](https://streamlit.io/cloud)
2. Logga in med GitHub
3. Klicka "New app"
4. Välj ditt repository
5. Main file: `app.py`
6. Klicka "Deploy"

Klart! 🎉

---

## 💡 Tips

- **Lokal utveckling**: Appen laddar om automatiskt när du sparar filer
- **Stoppa appen**: Tryck `Ctrl+C` i terminalen
- **Ändra port**: `streamlit run app.py --server.port 8502`
- **Debug**: Använd `st.write(variabel)` för att se värden

---

## ❓ Vanliga Problem

**"ModuleNotFoundError"**
→ Kör: `pip install -r requirements.txt`

**"Port already in use"**
→ Använd annan port: `streamlit run app.py --server.port 8502`

**"Database locked"**
→ Stäng alla Streamlit-instanser och starta om


