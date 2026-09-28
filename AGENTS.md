# Project direction

Organize the public curriculum and README around three connected layers:
1. Understand detailed LLM architecture, training steps and inference steps.
2. Implement those algorithms as readable, trainable and testable programs.
3. Deploy the same model onto our NPU and verify correctness and acceleration.
The A-H stages and C01-C40 units are subdivisions of these layers. NPU deployment
includes lowering, runtime, data loading, execution and readback, not only RTL blocks.

The primary purpose of llm-lab is deep, testable intuition about LLMs: representation,
learning, context use, generation, and execution cost. Code and hardware experiments
exist to strengthen that understanding. Preserve the complete algorithm-to-RTL path,
but do not let NPU implementation displace the model-learning curriculum.

Before extending curriculum or architecture, read docs/learning-plan.md,
docs/lesson-standard.md, and docs/learning-progress.md. For hardware work also read
docs/codesign-charter.md. The current user request takes precedence.

Use the same workload/model lineage across explanations, numerical references,
profiling and hardware experiments. Train/test separation, gradients, and a genuinely
trained small Transformer are priorities before claiming end-to-end LLM competence.
The existing TinyDecoder has random weights; the existing training lab is a bigram.

NanoNPU is a pinned reference only, not the project's specification, required build
dependency or architecture limit. Develop our choices from workload evidence.

Keep planned, implemented and executed work distinct. Python tests are not RTL
simulation evidence; cycle-model counts are not measured silicon performance.
Update the progress document when relevant milestones actually change.

For each substantive lesson include a concrete question, minimal worked example,
prediction-before-experiment, relevant counterexample and an understanding check.
Teach prerequisites where needed. Preserve existing lesson paths when reorganizing.
Do not add mandatory topics solely to increase breadth or chapter counts.
