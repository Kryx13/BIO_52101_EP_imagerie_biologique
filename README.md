# 🌿 KADSI – Application de préavis dermatologique assistée par IA

KADSI est un projet d’application mobile visant à proposer un *préavis dermatologique*, grâce à un VLM à partir d’une simple photo de lésion cutanée. L'application permet d'informer l'utilisateur s’il est pertinent de consulter un professionnel de santé, *sans diagnostic ni enregistrement d'image*, garantissant ainsi éthique et confidentialité.

---

## 📁 Contenu du Repository

| Dossier/Fichier             | Description |
|----------------------------|-------------|
| rapport_KADSI.pdf        | Rapport complet du projet, enjeux, méthode, résultats |
| vlm_grpo-main            | Scripts implémentant la méthode Grpo d'entrainement |
| build1_app.apk           | Premier prototype de l'application mobile KADSI |
| Fitzpatrick17.csv        | Dataset utilisé, contenant les métadonnées des images du Fitzpatrick17k |
|qwen_training-Copy1 (1).ipynb| Code pour entrainer le modèle
---

## 🔗 Présentation du projet

- Diaporama de présentation (Canva) :  
  👉 [Voir la présentation](https://www.canva.com/design/DAGnuaOHHt8/7KkiRshFJ6J1TmHxZ0P1Hw/edit?utm_content=DAGnuaOHHt8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## 💡 Objectifs du projet

- Réduire les *délais d’accès aux soins dermatologiques* en proposant un outil de préavis accessible.
- Offrir une solution inclusive, *sensible aux peaux foncées*, en intégrant des datasets diversifiés comme *Fitzpatrick17k*.
- Proposer une *interface simple, éthique et sécurisée*, pour une première analyse sans stockage ni profilage utilisateur.

---

## ⚙️ Technologies utilisées

- Python 3.10
- PyTorch / Transformers
- VLM-GRPO (libraire pour implémenter la méthode d'entrainement de deepseek à des modèles de visions (codé par nos soins))

---
## Entrainement

- Ci-dessous les logs d'entrainement pour la partie reinforcement learning

![Evolution du reward en fonction des steps](./images/reward.png)
![Evolution de la loss en fonction des steps](./images/loss.png)

NB : Il est tout à fait normal d’avoir une loss à 0 au début de l'entraînement. Cela est
dû à la manière dont la méthode Grpo de reinforcement learning est implémenté (il
dépend de la divergence du modèle entraîné par rapport au modèle de référence, or au
début, le modèle de référence est strictement égal au modèle entraîné).



## 👥 Équipe

- Kessel DIAROUMEYE 
- Sinoué GAD
- Ahmed IZI

---
