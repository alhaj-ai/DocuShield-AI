# Contributing to DocuShield AI

We welcome contributions from the community to improve **DocuShield AI**, extend Arabic & regional PII patterns, or optimize multimodal inference!

## How to Contribute

1. **Fork the Repository**:
   Click the Fork button at the top right of this page.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/DocuShield-AI.git
   cd DocuShield-AI
   ```
3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Make your changes**:
   - Write clean, type-annotated code.
   - Ensure all tests pass: `python tests/test_arabic_pii.py` and `python tests/test_api.py`.
5. **Commit your changes**:
   ```bash
   git commit -m "feat: add support for new regional PII detector"
   ```
6. **Push to branch and open a Pull Request**.

## Areas for Contribution
- Additional regional identification formats (GCC, North Africa, Levant).
- Quantization & TensorRT / ONNX inference export.
- Frontend enhancements for the Streamlit dashboard.
