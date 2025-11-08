# LM Studio Setup Guide

LM Studio on käyttäjäystävällinen graafinen työkalu paikallisten LLM-mallien ajamiseen. Se tarjoaa yksinkertaisen tavan käyttää avoimen lähdekoodin malleja ilman komentorivikokemusta.

## Asennus

1. Lataa LM Studio: https://lmstudio.ai
2. Asenna sovellus
3. Käynnistä LM Studio

## Mallin lataaminen

1. Avaa **"Search"** -välilehti
2. Etsi haluamasi malli (suositukset alla)
3. Klikkaa **Download**
4. Odota latauksen valmistuminen

### Suositellut mallit suomenkieliseen RAG-järjestelmään:

**Mistral 7B Instruct** (Suositus)
- Hyvä tasapaino laadun ja nopeuden välillä
- ~4GB levytilaa
- Toimii hyvin suomeksi
- Etsi: `mistral-7b-instruct`

**Llama 3.1 8B Instruct**
- Erinomainen laatu
- ~4.7GB levytilaa
- Hyvä monikielisyys
- Etsi: `llama-3.1-8b-instruct`

**Phi-3 Medium** (Kevyt vaihtoehto)
- Pienempi malli, nopeampi
- ~2.4GB levytilaa
- Riittävä yksinkertaisiin kysymyksiin
- Etsi: `phi-3-medium`

## Palvelimen käynnistys

1. Siirry **"Local Server"** -välilehdelle
2. Valitse ladattu malli pudotusvalikosta
3. Klikkaa **"Start Server"**
4. Palvelin käynnistyy osoitteeseen: `http://localhost:1234`

### Tärkeät asetukset:

- **Port**: 1234 (oletus, voit vaihtaa tarvittaessa)
- **CORS**: Enable (jos käytät selaimesta)
- **Context Length**: 4096 tai suurempi (RAG-järjestelmälle)
- **Temperature**: 0.3 (asetetaan automaattisesti koodissa)

## Konfigurointi drive-rag järjestelmään

Muokkaa `.env` tiedostoa:

```env
# Aseta provider openai:ksi
LLM_PROVIDER=openai

# LM Studio asetukset
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=local-model
```

**Huom**: `OPENAI_MODEL` arvon tulee olla `local-model` tai mallin nimi joka näkyy LM Studiossa.

## Testaaminen

Testaa että LM Studio toimii:

```bash
# Testaa suoraan LM Studio API:a
curl http://localhost:1234/v1/models

# Testaa drive-rag integraatiota
python scripts/test_llm_provider.py
```

## Ongelmatilanteita

### "Connection refused"
- Varmista että LM Studio palvelin on käynnissä
- Tarkista että portti on 1234 (tai päivitä .env)
- Tarkista palomuuri-asetukset

### Hidas vastausaika
- Käytä pienempää mallia (esim. Phi-3)
- Varmista että GPU-kiihdytys on käytössä LM Studiossa
- Pienennä context length asetusta

### "Model not found"
- Varmista että malli on ladattu kokonaan
- Tarkista mallin nimi LM Studion "Local Server" välilehdeltä
- Päivitä `OPENAI_MODEL` .env tiedostossa

### Epäjohdonmukaiset vastaukset
- Nosta context length arvoa (min. 4096)
- Tarkista että temperature on sopiva (0.3 RAG:lle)
- Kokeile eri mallia

## GPU-kiihdytys

LM Studio käyttää automaattisesti GPU:ta jos se on saatavilla:

- **NVIDIA**: CUDA tuki sisäänrakennettu
- **AMD**: ROCm tuki uudemmissa versioissa
- **Apple Silicon**: Metal tuki M1/M2/M3 prosessoreille

Tarkista GPU-käyttö LM Studion alaosasta käynnistyksen jälkeen.

## Edut verrattuna Ollamaan

**LM Studio:**
✓ Graafinen käyttöliittymä
✓ Helppo mallien lataaminen ja hallinta
✓ Sisäänrakennettu chat-testausympäristö
✓ Mallin asetusten helppo säätäminen
✓ Windows/Mac native sovellus

**Ollama:**
✓ Kevyempi ja nopeampi
✓ Parempi komentorivikäyttöön
✓ Helpompi automatisointiin
✓ Vähemmän resursseja

## Lisätietoja

- Virallinen dokumentaatio: https://lmstudio.ai/docs
- Mallit: https://huggingface.co/models
- Tuki: https://lmstudio.ai/discord
