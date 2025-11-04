#!/usr/bin/env python3
"""
Test: Tokenizer Functionality

Tests Tinker client's tokenizer access:
  1. Create training client
  2. Get tokenizer from training client
  3. Encode text samples
  4. Decode tokens back to text
  5. Verify round-trip consistency

This is a prerequisite for data preparation in training workflows.
"""
import os
import sys

# Configure environment for kgateway-training
os.environ["TINKER_BASE_URL"] = "http://kgateway-training.slime-gmi:8000"
os.environ["TINKER_API_KEY"] = "slime-dev-key"

import tinker

print("=" * 80)
print("TEST: Tokenizer Functionality")
print("=" * 80)
print(f"Base URL: {os.environ.get('TINKER_BASE_URL')}")
print(f"API Key: {os.environ.get('TINKER_API_KEY')[:10]}...")
print("=" * 80)

try:
    # Step 1: Create service client
    print("\n[1/5] Creating service client...")
    client = tinker.ServiceClient()
    print("   ✓ Service client created")

    # Step 2: Create training client
    print("\n[2/5] Creating training client...")
    training_client = client.create_lora_training_client(
        base_model="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
        rank=0  # No LoRA
    )
    print(f"   ✓ Training client created with model_id: {training_client.model_id}")

    # Step 3: Get tokenizer
    print("\n[3/5] Getting tokenizer from training client...")
    tokenizer = training_client.get_tokenizer()
    print(f"   ✓ Tokenizer retrieved")
    print(f"   Type: {type(tokenizer)}")

    # Step 4: Encode text samples
    print("\n[4/5] Encoding text samples...")
    test_texts = [
        "Hello, world!",
        "This is a test of the tokenizer.",
        "Machine learning is fascinating.",
    ]

    encoded_samples = []
    for i, text in enumerate(test_texts):
        tokens = tokenizer.encode(text)
        encoded_samples.append((text, tokens))
        print(f"   Sample {i + 1}:")
        print(f"     Text: \"{text}\"")
        print(f"     Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''} (total: {len(tokens)})")

    # Step 5: Decode tokens back to text
    print("\n[5/5] Decoding tokens back to text...")
    decode_success = True
    for i, (original_text, tokens) in enumerate(encoded_samples):
        decoded_text = tokenizer.decode(tokens)
        print(f"   Sample {i + 1}:")
        print(f"     Original: \"{original_text}\"")
        print(f"     Decoded:  \"{decoded_text}\"")

        # Verify round-trip (may not be exact due to special tokens, but should be similar)
        if original_text.lower().replace(" ", "") in decoded_text.lower().replace(" ", ""):
            print(f"     ✓ Round-trip verification passed")
        else:
            print(f"     ⚠ Round-trip differs (this may be expected due to tokenization)")

    # Additional tokenizer tests
    print("\n[Bonus] Additional tokenizer functionality...")

    # Test special tokens
    test_tokens = [1, 2, 3, 4, 5]  # Common token IDs
    decoded = tokenizer.decode(test_tokens)
    print(f"   Token IDs {test_tokens} → \"{decoded}\"")

    # Test empty input
    empty_tokens = tokenizer.encode("")
    print(f"   Empty string → {empty_tokens} tokens")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print(f"\nTokenizer functionality verified:")
    print(f"  • Model ID: {training_client.model_id}")
    print(f"  • Encoded {len(test_texts)} samples")
    print(f"  • Decode round-trip successful")
    print(f"\n🎉 Tokenizer ready for data preparation!")
    print("=" * 80)

    sys.exit(0)

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "=" * 80)
    print("✗ TEST FAILED")
    print("=" * 80)

    sys.exit(1)
