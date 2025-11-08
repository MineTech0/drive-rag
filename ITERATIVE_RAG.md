# Iterative Agentic RAG

## Mikä se on?

Iteratiivinen agentti joka **hakee niin kauan kunnes on tyytyväinen** tuloksen laatuun. Tavallisen RAG:n sijasta, joka tekee yhden haun, tämä agentti:

1. **Arvioi** - Onko löydetty tieto riittävää?
2. **Tunnistaa puutteet** - Mitä vielä tarvitaan?
3. **Hakee uudelleen** - Muotoilee uusia kyselyjä puuttuvalle tiedolle
4. **Jatkaa** - Kunnes luottamustaso >= 85% tai max 5 kierrosta

## Milloin käyttää?

### Käytä `/ask-iterative` kun:
- ✅ "Etsi **KAIKKI** tanssikerhon puheenjohtajat"
- ✅ "Hae **kattavasti** tietoa projektista"
- ✅ "Kerro **kaikki** mitä tiedät aiheesta"
- ✅ "Listaa **jokainen** kokous vuodelta 2020"
- ✅ Haluat maksimaalisen määrän lähteitä (jopa 100 kpl)
- ✅ Tärkeä kysymys jossa ei saa puuttua tietoa

### Käytä tavallista `/ask` kun:
- ❌ Nopea kysymys
- ❌ Tarvitset vain muutaman lähteen (6-8 kpl)
- ❌ Yksinkertainen fakta

## Käyttö

### cURL esimerkki:

```bash
curl -X POST http://localhost:8000/ask-iterative \
  -H "Content-Type: application/json" \
  -d '{
    "query": "etsi kaikki tanssikerhon puheenjohtajat eri vuosilta",
    "multi_query": true
  }'
```

### Python esimerkki:

```python
import requests

response = requests.post(
    "http://localhost:8000/ask-iterative",
    json={
        "query": "hae kattavasti tietoa projektin vaiheista",
        "multi_query": True
    }
)

result = response.json()

print(f"Vastaus ({result['total_iterations']} kierrosta):")
print(result['answer'])
print(f"\nLähteet: {result['total_sources']} kpl")
print(f"Lopullinen varmuus: {result['final_confidence']:.0%}")

# Näytä iteraatiohistoria
for iteration in result['iterations']:
    print(f"\nKierros {iteration['iteration']}:")
    print(f"  Kysely: {iteration['query']}")
    print(f"  Tuloksia: {iteration['num_results']}")
    print(f"  Varmuus: {iteration['confidence']:.0%}")
    print(f"  Arvio: {iteration['assessment']}")
```

## Vastausformaatti

```json
{
  "answer": "Kattava vastaus käyttäen kaikkia löydettyjä lähteitä...",
  "sources": [
    {
      "file_name": "kokous_2015.pdf",
      "link": "https://drive.google.com/...",
      "locator": "sivu 3",
      "chunk_id": "abc123",
      "snippet": "...tekstinäyte..."
    }
  ],
  "latency_ms": 8500,
  "iterations": [
    {
      "iteration": 1,
      "query": "etsi kaikki tanssikerhon puheenjohtajat",
      "num_results": 20,
      "assessment": "Löydettiin tietoa vain vuosilta 2015-2022",
      "confidence": 0.65,
      "missing_info": ["Vuodet 2023-2024", "Aiemmat vuodet"]
    },
    {
      "iteration": 2,
      "query": "tanssikerhon puheenjohtaja 2023 2024",
      "num_results": 25,
      "assessment": "Kattava kuva saavutettu",
      "confidence": 0.90,
      "missing_info": []
    }
  ],
  "total_sources": 25,
  "total_iterations": 2,
  "final_confidence": 0.90
}
```

## Asetukset

Agentin asetukset (kovakoodattu, voidaan myöhemmin tehdä konfiguroitavaksi):

- **max_iterations**: 5 kierrosta
- **confidence_threshold**: 0.85 (85% varmuus)
- **max_sources**: 100 lähdettä
- **initial_candidates**: 100 hakutulosta per kierros

## Miten se toimii sisäisesti?

### Kierros 1:
```
Käyttäjä: "etsi kaikki puheenjohtajat"
→ Hae 100 kandidaattia
→ Rerankkaa parhaat 30
→ LLM arvioi: "Löytyi 2015-2022, puuttuu 2023+"
→ Varmuus: 65% → JATKA
```

### Kierros 2:
```
LLM generoi: "tanssikerhon puheenjohtaja 2023 2024"
→ Hae 100 uutta kandidaattia
→ Yhdistä aiempiin (deduplikointi)
→ Rerankkaa KAIKKI (nyt ~40 uniikkia)
→ LLM arvioi: "Nyt kattava!"
→ Varmuus: 90% → PYSÄHDY
```

### Lopullinen vastaus:
```
→ LLM saa KAIKKI 30 parasta lähdettä
→ Erikoisprompti: "Käytä KAIKKIA lähteitä, älä jätä mitään pois"
→ Generoi 2000 tokenin kattava vastaus
→ Palauta käyttäjälle
```

## Vertailu

| Ominaisuus | `/ask` | `/ask-iterative` |
|-----------|--------|------------------|
| Hakukierroksia | 1 | 1-5 (adaptiivinen) |
| Lähteitä | 6-20 | 10-30 |
| Nopeus | ~2-3s | ~5-15s |
| Kattavuus | Hyvä | Erinomainen |
| Käyttötapaus | Pikakysymykset | Kattavat selvitykset |

## Tulevat parannukset

1. **Konfiguroitavat parametrit** - Käyttäjä voi säätää max_iterations, threshold jne.
2. **Streaming** - Näytä edistyminen reaaliajassa
3. **Välimuisti** - Älä hae samoja dokumentteja uudestaan
4. **Parallelointi** - Hae useammalla kyselyllä yhtäaikaa
5. **Metriikka** - Tallenna onnistumisprosentit ja optimoi

## Vinkit

**Paras tapa käyttää:**
```bash
# Aloita aina laajalla kyselyllä
"etsi kaikki X"

# Ei näin:
"kuka oli Y vuonna 2015?"  # Liian spesifi, löytää vain yhden
```

**Optimaaliset kysymysmuodot:**
- "Etsi kaikki..."
- "Hae kattavasti..."
- "Listaa jokainen..."
- "Kerro kaikki mitä tiedät..."
- "Mitkä kaikki..."

Nämä laukaisevat sekä:
1. Exhaustive search -tilan (100 kandidaattia)
2. Iteratiivisen agentti-logiikan (jatkaa kunnes tyytyväinen)

## Huomioita

- **Hitaampi** kuin tavallinen `/ask` (5-15s vs 2-3s)
- **Kalliimpi** LLM-kutsuja (arviointi + vastaus per kierros)
- **Tarkempi** - Vähemmän todennäköisesti jättää tietoa pois
- **Läpinäkyvä** - Näet jokaisen kierroksen reasoning:in

---

**TL;DR**: Käytä `/ask-iterative` kun haluat VARMASTI kaiken tiedon. Agentti ei lopeta ennen kuin on varma että löysi kaiken oleellisen!
