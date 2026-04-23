# Doom Index v2 — 7-Day Execution Plan

**Your goal:** Get from current state to viva-ready multimodal system.
**Your resources:** Multiple H100s, 1 week, this codebase.

---

## Day 1 (Today): Foundation

**Morning (4 hours)**
1. Run migration: `python migrate.py --force`
2. Install dependencies: `pip install -r requirements.txt`
3. Run validation: `python validate_setup.py`
4. Fix any issues

**Afternoon (4 hours)**
5. Process your Pushshift data:
   ```bash
   python train_model_full_fixed.py \
       --data_dir "doom index/data/twitter_dataset/scraped_data/reddit" \
       --output data/processed_reddit_multimodal.csv \
       --max_files 50
   ```
6. Verify output: `head data/processed_reddit_multimodal.csv`
7. Check label distribution — should be ~20-30% positive

**Evening (2 hours)**
8. Build Neo4j graph:
   ```bash
   docker-compose up neo4j -d
   python build_neo4j_graph.py --data_path data/processed_reddit_multimodal.csv
   ```
9. Verify graph: open http://localhost:7474

**Deliverable:** `data/processed_reddit_multimodal.csv` with proper labels + populated Neo4j

---

## Day 2: First Training Run

**Morning (3 hours)**
1. Pre-download model weights (if HPC has no internet):
   ```bash
   python download_models.py
   scp -r ~/.cache/huggingface vivek.120542@10.16.1.50:~/.cache/
   ```

2. Submit first training job:
   ```bash
   qsub hpc_multimodal_train.sh
   ```

**Afternoon (4 hours)**
3. While training runs, work on API v2 integration:
   - Read `api_v2.py`
   - Understand the endpoints
   - Test locally with small model

4. Start dashboard development:
   ```bash
   streamlit run dashboard/app.py
   ```

**Evening (3 hours)**
5. Monitor training logs: `tail -f logs/doom_multimodal_train.log`
6. If training fails, debug and resubmit
7. Read through `INTEGRATION_GUIDE.md` for troubleshooting

**Deliverable:** First trained model checkpoint OR debugged training pipeline

---

## Day 3: Model Training + API Integration

**Morning (4 hours)**
1. Training should complete overnight. Check results:
   ```bash
   ls models/multimodal_doom/
   ```
2. Evaluate model:
   ```bash
   python evaluate_model.py \
       --model_path models/multimodal_doom/best_model.pt \
       --data_path data/processed_reddit_multimodal.csv
   ```
3. Check metrics — target: accuracy > 85%, F1 > 0.80

**Afternoon (4 hours)**
4. Integrate model into API:
   ```bash
   python api_v2.py
   ```
5. Test endpoints:
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "Test post", "author_id": "test"}'
   ```

**Evening (2 hours)**
6. Test attack simulator endpoint
7. Fix any API bugs

**Deliverable:** Working API with /analyze and /attack endpoints

---

## Day 4: Dashboard + Attack Simulator

**Morning (4 hours)**
1. Polish Streamlit dashboard:
   - Test all 4 tabs
   - Ensure plots render correctly
   - Fix any UI issues

2. Test attack simulator thoroughly:
   - Try different input texts
   - Verify toxicity budget enforcement
   - Check doom uplift calculations

**Afternoon (4 hours)**
3. Implement privacy modules:
   ```bash
   # Test DP training (small scale)
   python -c "from src.privacy.dp_trainer import DPDoomTrainer; print('DP OK')"

   # Test FL simulation
   python -c "from src.privacy.fl_simulator import FLSimulator; print('FL OK')"
   ```

4. Generate privacy-utility tradeoff data (can be mocked for viva)

**Evening (2 hours)**
5. Generate viva plots:
   ```bash
   python generate_viva_plots.py --output_dir viva_plots
   ```
6. Review all plots for presentation quality

**Deliverable:** Complete dashboard + attack simulator + viva plots

---

## Day 5: Integration + Testing

**Morning (4 hours)**
1. Full system integration test:
   ```bash
   # Terminal 1
   python api_v2.py

   # Terminal 2
   streamlit run dashboard/app.py

   # Terminal 3
   python demo.py
   ```

2. Run unit tests:
   ```bash
   pytest tests/test_multimodal.py -v
   ```

**Afternoon (4 hours)**
3. Docker test:
   ```bash
   docker-compose up --build -d
   ```
4. Verify all services start correctly
5. Test API through Docker

**Evening (2 hours)**
6. Run complete demo script: `python demo.py`
7. Time the full flow — should be < 5 minutes
8. Fix any integration issues

**Deliverable:** Fully integrated system running end-to-end

---

## Day 6: Viva Prep

**Morning (4 hours)**
1. Create presentation slides:
   - Slide 1: Title + architecture diagram
   - Slide 2: Data pipeline (Pushshift -> weak labels)
   - Slide 3: Model architecture (GraphSAGE + DistilBERT)
   - Slide 4: Training results (metrics, confusion matrix, ROC)
   - Slide 5: Attack simulator demo
   - Slide 6: Privacy analysis (DP + FL)
   - Slide 7: Dashboard walkthrough
   - Slide 8: Future work

2. Insert generated plots into slides

**Afternoon (4 hours)**
3. Practice demo flow:
   - Safe post -> LOW risk
   - Controversial post -> CRITICAL risk
   - Attack simulation -> show variants
   - Privacy tab -> show tradeoff curves

4. Prepare for questions:
   - "Why weak labels?" -> Explain heuristic rationale
   - "Why GraphSAGE?" -> Explain inductive capability
   - "Why DistilBERT?" -> Balance of speed and accuracy
   - "Ethical concerns?" -> DP + FL + only public data

**Evening (2 hours)**
5. Backup everything:
   ```bash
   git add .
   git commit -m "v2 complete - viva ready"
   git push origin v2-multimodal
   ```
6. Test on clean environment (optional)

**Deliverable:** Presentation slides + practiced demo

---

## Day 7: Final Polish + Viva

**Morning (3 hours)**
1. Final system check:
   ```bash
   python validate_setup.py
   ```
2. Start all services:
   ```bash
   docker-compose up -d
   ```
3. Run one final demo: `python demo.py`

**Afternoon: VIVA**
4. Arrive early, test projector/screen sharing
5. Have backup plan: screenshots if live demo fails
6. Stay calm, you've built something impressive

---

## 🚨 Critical Path (If You Fall Behind)

**Minimum viable viva:**
1. ✅ Properly labeled data (Day 1)
2. ✅ Trained multimodal model (Day 2-3)
3. ✅ Working API with /analyze (Day 3)
4. ✅ Streamlit dashboard (Day 4)
5. ✅ Demo script runs (Day 5)

**Drop these if time runs out:**
- Docker deployment (nice to have)
- Full FL simulation (mock data is fine)
- ONNX export (optional optimization)
- Extensive hyperparameter tuning

**Never drop these:**
- Proper labels (your scientific credibility)
- Working model (the core deliverable)
- Demo flow (what the examiner sees)

---

## 📞 Emergency Commands

```bash
# Training crashed? Restart quickly:
qsub hpc_multimodal_train.sh

# API won't start? Check model path:
ls models/multimodal_doom/best_model.pt

# Neo4j down? Restart:
docker-compose restart neo4j

# Out of disk space?
df -h
rm -rf logs/*.log backup_*/

# Need to debug training?
tail -n 100 logs/doom_multimodal_train.log

# Dashboard won't load?
streamlit run dashboard/app.py --server.port 8502
```

---

## 💡 Pro Tips

1. **Sleep > Code.** A tired brain writes bugs. 6 hours sleep minimum.

2. **Commit often.** `git commit -m "wip"` every 2 hours.

3. **Test on HPC early.** Don't wait until Day 6 to discover CUDA issues.

4. **Have screenshots.** If live demo fails, show screenshots of it working.

5. **Know your numbers.** Memorize: accuracy, F1, AUC, model size, training time.

6. **The attack simulator is your ace.** It's unique, visual, and memorable.

7. **If asked about ethics:** Emphasize DP, FL, public data only, and the mental health angle (warning at-risk users).

---

**You've got the H100s, you've got the code, you've got the plan.**

**Execute. Dominate. Graduate.** 🔥
