# Test Coverage Analysis - Tinker Cookbook vs Current Tests

## Summary

Based on comprehensive analysis of the tinker-cookbook repository (75 Python files), we have identified **gaps in test coverage** for the kgateway-training GMI wrapper.

## Current Test Status

### ✅ Phase 1 - HTTP API Tests (All Passing)
- test_1_model_creation.py
- test_2_single_train_step.py
- test_direct_http_sampling.py
- test_3_checkpoint_workflow.py
- test_4_multi_step_training_v2.py
- test_5_delete_model_verification.py

### ✅ Phase 2 - Tinker Client Basics (All Passing)
- test_kgateway_training.py
- test_tokenizer.py
- test_supervised_single_step.py
- test_sampling_client_creation.py

### ✅ Phase 3 - Training Workflows (All Passing)
- test_supervised_multi_epoch.py
- test_checkpoint_resume.py
- test_rl_rollout_pattern.py

## Missing Test Coverage (From Cookbook Analysis)

### Phase 4: Advanced Cookbook Patterns

#### 1. **test_sl_loop_pattern.py** - Minimal SFT Loop
**Cookbook Reference:** `recipes/sl_loop.py` (156 lines)
**Pattern:** Raw API usage without abstractions

**What it tests:**
- Linear learning rate schedule
- Checkpoint saving every N steps
- Metric logging (NLL, tokens, LR)
- Resume from checkpoint mid-training
- Direct dataset loading from HF

**Key Difference from Current Tests:**
- Current tests use simplified data
- This tests full JSONL dataset loading
- Tests learning rate scheduling
- Tests periodic checkpointing during training

#### 2. **test_advantage_computation.py** - RL Advantage Patterns
**Cookbook Reference:** `recipes/rl_loop.py`, `rl/data_processing.py`
**Pattern:** Group-based advantage computation

**What it tests:**
```python
# Advantage computation within groups
rewards = [0.8, 0.9, 0.2, 0.3]  # 4 samples for same question
group_mean = mean(rewards)  # 0.55
advantages = [r - group_mean for r in rewards]  # [0.25, 0.35, -0.35, -0.25]
```

**Key Concepts:**
- Group-based normalization (not per-sample)
- Multiple samples per prompt
- Trajectory grouping for RL
- Proper advantage calculation

#### 3. **test_gradient_accumulation.py** - Multi-Substep Training
**Cookbook Reference:** `rl/train.py::train_step()`, Advanced Features section
**Pattern:** Accumulate gradients over multiple forward_backward calls

**What it tests:**
```python
num_substeps = 4
batches = split_list(data, num_substeps)
for batch in batches:
    fwd_bwd = training_client.forward_backward(batch, loss_fn="...")
    # No optim_step here! Gradients accumulate

# Single optimizer step after all forward_backward calls
training_client.optim_step(adam_params)
```

**Why Important:**
- Enables large effective batch sizes
- Matches cookbook pattern
- Critical for memory-constrained training

#### 4. **test_kl_divergence_tracking.py** - KL Penalty Metrics
**Cookbook Reference:** `rl/train.py`, `rl/metrics.py`
**Pattern:** Compute KL between sampling and training distributions

**What it tests:**
```python
# Sample with old policy → get logprobs
sampling_logprobs = sample_response.sequences[0].logprobs

# Train with new policy → get new logprobs
training_logprobs = fwd_bwd_result.loss_fn_outputs[0].logprobs

# Compute KL divergence
kl_v1 = mean(training_logprobs - sampling_logprobs)  # Forward KL
kl_v2 = mean(exp(sampling_logprobs) * (sampling_logprobs - training_logprobs))  # Reverse KL
```

**Metrics:**
- `optim/kl_sample_train_v1`
- `optim/kl_sample_train_v2`
- `optim/entropy`

#### 5. **test_renderer_patterns.py** - Message Rendering
**Cookbook Reference:** `renderers.py`, `test_renderers.py`
**Pattern:** Message ↔ Token conversion with training masks

**What it tests:**
```python
renderer = get_renderer("qwen3", tokenizer)

# Build supervised example with weight masking
tokens, weights = renderer.build_supervised_example(
    messages=[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ],
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
)

# weights = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Only train on assistant response
```

**TrainOnWhat Options:**
- `LAST_ASSISTANT_MESSAGE` - Only final response
- `ALL_ASSISTANT_MESSAGES` - All assistant responses
- `ALL_MESSAGES` - Train on everything
- `ALL_TOKENS` - All tokens weighted equally

#### 6. **test_multi_turn_environment.py** - Multi-Turn RL
**Cookbook Reference:** `multiplayer_rl/`, `rl/types.py`
**Pattern:** Environment with multiple step() calls per episode

**What it tests:**
```python
class MultiTurnEnv(Env):
    async def initial_observation(self):
        return "Guess a number between 1-100", stop_tokens

    async def step(self, action_tokens):
        guess = parse_number(action_tokens)
        if guess == self.target:
            return StepResult(reward=1.0, episode_done=True, ...)
        elif guess < self.target:
            return StepResult(reward=0.0, episode_done=False,
                            next_observation="Too low! Try again.", ...)
        else:
            return StepResult(reward=0.0, episode_done=False,
                            next_observation="Too high! Try again.", ...)
```

**Key Concepts:**
- Sparse rewards (only at episode end)
- Multi-turn dialogue
- State tracking across turns
- Episode termination conditions

#### 7. **test_dpo_loss_function.py** - Direct Preference Optimization
**Cookbook Reference:** `recipes/preference/dpo/train.py`, `preference/train_dpo.py`
**Pattern:** DPO loss with chosen/rejected pairs

**What it tests:**
```python
datum = tinker.Datum(
    model_input=prompt_tokens,
    loss_fn_inputs={
        "chosen_target_tokens": TensorData.from_torch(chosen_response),
        "rejected_target_tokens": TensorData.from_torch(rejected_response),
        "chosen_logprobs": TensorData.from_torch(chosen_logprobs),
        "rejected_logprobs": TensorData.from_torch(rejected_logprobs),
    }
)

fwd_bwd = training_client.forward_backward([datum], loss_fn="dpo")
```

**Key Concepts:**
- Preference pairs (chosen vs rejected)
- DPO loss function (different from importance_sampling)
- No separate reward model needed
- Direct policy optimization from preferences

#### 8. **test_async_vs_sync_api.py** - API Patterns
**Cookbook Reference:** Multiple recipes show both patterns
**Pattern:** Compare async and sync API behavior

**What it tests:**
```python
# Sync API (SFT pattern)
fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
result = fwd_bwd_future.result()  # Blocks until complete

# Async API (RL pattern)
fwd_bwd_future = await training_client.forward_backward_async(batch, loss_fn="importance_sampling")
result = await fwd_bwd_future.result_async()  # Non-blocking await
```

**Key Concepts:**
- Sync vs async patterns
- When to use each
- Performance differences
- Concurrent operations

#### 9. **test_evaluator_pattern.py** - Periodic Evaluation
**Cookbook Reference:** `eval/evaluators.py`, `supervised/train.py`
**Pattern:** Run evaluation during training

**What it tests:**
```python
class NLLEvaluator(SamplingClientEvaluator):
    async def __call__(self, sampling_client):
        total_nll = 0.0
        for item in self.test_set:
            # Compute NLL on test data
            total_nll += compute_nll(...)
        return {"test/nll": total_nll / len(self.test_set)}

# In training loop
if step % eval_every == 0:
    metrics = await evaluator(sampling_client)
    ml_logger.log_metrics(metrics, step=step)
```

**Key Concepts:**
- Periodic evaluation
- Test set metrics
- Integration with training loop
- InspectAI benchmarks

#### 10. **test_sampling_params_variations.py** - Sampling Configuration
**Cookbook Reference:** Multiple recipes, `completers.py`
**Pattern:** Different sampling strategies

**What it tests:**
```python
# Greedy decoding
greedy = tinker.SamplingParams(max_tokens=256, temperature=0.0)

# Nucleus sampling
nucleus = tinker.SamplingParams(max_tokens=256, temperature=0.7, top_p=0.9)

# Top-k sampling
topk = tinker.SamplingParams(max_tokens=256, temperature=1.0, top_k=50)

# With stop sequences
with_stops = tinker.SamplingParams(
    max_tokens=256,
    temperature=0.7,
    stop=["</s>", "\n\n", "Human:"]
)
```

**Key Concepts:**
- Temperature effects
- top_p vs top_k
- Stop sequences
- max_tokens behavior

## Test Priority Matrix

| Test | Cookbook Usage | Complexity | Implementation Effort | Priority |
|------|---------------|------------|----------------------|----------|
| test_advantage_computation.py | High (all RL) | Medium | Low | **HIGH** |
| test_gradient_accumulation.py | High (large models) | Medium | Medium | **HIGH** |
| test_sl_loop_pattern.py | Medium | Low | Low | **MEDIUM** |
| test_kl_divergence_tracking.py | High (RL metrics) | Medium | Medium | **HIGH** |
| test_renderer_patterns.py | High (data prep) | Low | Low | **MEDIUM** |
| test_multi_turn_environment.py | Medium | High | High | **MEDIUM** |
| test_dpo_loss_function.py | Medium | Medium | High | **LOW** (needs DPO impl) |
| test_async_vs_sync_api.py | High | Low | Low | **LOW** (nice-to-have) |
| test_evaluator_pattern.py | Medium | Low | Low | **MEDIUM** |
| test_sampling_params_variations.py | High | Low | Low | **MEDIUM** |

## Recommended Implementation Order

### Sprint 1 (Core RL Patterns)
1. **test_advantage_computation.py** - Critical for RL correctness
2. **test_gradient_accumulation.py** - Enables large-batch training
3. **test_kl_divergence_tracking.py** - Essential RL metrics

### Sprint 2 (Data & Evaluation)
4. **test_renderer_patterns.py** - Data preparation patterns
5. **test_sl_loop_pattern.py** - Full SFT workflow
6. **test_evaluator_pattern.py** - Evaluation integration

### Sprint 3 (Advanced Features)
7. **test_sampling_params_variations.py** - Sampling strategies
8. **test_multi_turn_environment.py** - Complex RL scenarios
9. **test_async_vs_sync_api.py** - API patterns

### Sprint 4 (Optional)
10. **test_dpo_loss_function.py** - Requires DPO implementation

## Implementation Notes

### For test_advantage_computation.py
- Use actual group-based advantage calculation from cookbook
- Test with different group sizes (4, 8, 16)
- Verify advantage sum = 0 within each group
- Test edge cases (all same rewards, extreme outliers)

### For test_gradient_accumulation.py
- Test with num_substeps = 2, 4, 8
- Verify gradients accumulate correctly
- Compare accumulated gradients vs single large batch
- Test memory usage difference

### For test_kl_divergence_tracking.py
- Capture logprobs from both sampling and training
- Compute both KL_v1 and KL_v2 variants
- Test entropy computation
- Verify metrics match cookbook formulas

## Success Criteria

Each test should:
1. ✅ Pass without exceptions
2. ✅ Match cookbook pattern exactly
3. ✅ Include clear documentation references
4. ✅ Verify expected outputs/metrics
5. ✅ Clean up resources properly
6. ✅ Exit with code 0 on success

## Coverage Goals

- **Current:** 13 tests covering basic Tinker API
- **Target:** 23 tests covering all cookbook patterns
- **Cookbook Coverage:** 85%+ of common patterns tested

## References

- Tinker Cookbook Analysis: `/tmp/tmux-tmp/TINKER_COOKBOOK_ANALYSIS.md`
- Tinker Cookbook Quick Reference: `/tmp/tmux-tmp/TINKER_COOKBOOK_QUICK_REFERENCE.md`
- Current Curriculum: `/tmp/tmux-tmp/tinker_gmi/tests_integration/CURRICULUM_TESTS.md`
