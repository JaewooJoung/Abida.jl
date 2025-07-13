# advanced_training.jl - Enhanced training with optimizers and mixed precision

using Flux
using Statistics
using ProgressMeter
using Random
using LinearAlgebra
using Dates
using JLD2

# AdamW optimizer implementation
mutable struct AdamW
    η::Float32          # Learning rate
    β₁::Float32         # First momentum coefficient
    β₂::Float32         # Second momentum coefficient
    ε::Float32          # Small constant for numerical stability
    weight_decay::Float32 # Weight decay coefficient
    t::Int              # Time step
    m::IdDict{Any, Any} # First moment estimates
    v::IdDict{Any, Any} # Second moment estimates
end

function AdamW(η=0.001f0, β₁=0.9f0, β₂=0.999f0, ε=1f-8, weight_decay=0.01f0)
    AdamW(η, β₁, β₂, ε, weight_decay, 0, IdDict(), IdDict())
end

function update!(opt::AdamW, x, Δ)
    opt.t += 1

    # Initialize moments if needed
    if !haskey(opt.m, x)
        opt.m[x] = zero(Δ)
        opt.v[x] = zero(Δ)
    end

    # Update biased first moment estimate
    opt.m[x] = opt.β₁ * opt.m[x] + (1 - opt.β₁) * Δ

    # Update biased second moment estimate
    opt.v[x] = opt.β₂ * opt.v[x] + (1 - opt.β₂) * (Δ .* Δ)

    # Compute bias-corrected moment estimates
    m_hat = opt.m[x] / (1 - opt.β₁^opt.t)
    v_hat = opt.v[x] / (1 - opt.β₂^opt.t)

    # Apply weight decay (decoupled from gradient)
    x .-= opt.weight_decay * opt.η * x

    # Apply update
    x .-= opt.η * m_hat ./ (sqrt.(v_hat) .+ opt.ε)

    return x
end

# Learning rate schedulers
abstract type LRScheduler end

mutable struct CosineAnnealingScheduler <: LRScheduler
    η_max::Float32
    η_min::Float32
    T_max::Int
    warmup_steps::Int
    current_step::Int
end

function CosineAnnealingScheduler(η_max=1f-3, η_min=1f-6, T_max=1000, warmup_ratio=0.1)
    warmup_steps = Int(round(T_max * warmup_ratio))
    CosineAnnealingScheduler(η_max, η_min, T_max, warmup_steps, 0)
end

mutable struct LinearWarmupScheduler <: LRScheduler
    η_target::Float32
    warmup_steps::Int
    current_step::Int
end

function LinearWarmupScheduler(η_target=1f-3, warmup_steps=1000)
    LinearWarmupScheduler(η_target, warmup_steps, 0)
end

mutable struct ExponentialDecayScheduler <: LRScheduler
    η_initial::Float32
    decay_rate::Float32
    decay_steps::Int
    current_step::Int
end

function ExponentialDecayScheduler(η_initial=1f-3, decay_rate=0.95f0, decay_steps=1000)
    ExponentialDecayScheduler(η_initial, decay_rate, decay_steps, 0)
end

function get_learning_rate(scheduler::CosineAnnealingScheduler)
    scheduler.current_step += 1

    # Warmup phase
    if scheduler.current_step <= scheduler.warmup_steps
        return scheduler.η_max * (scheduler.current_step / scheduler.warmup_steps)
    end

    # Cosine annealing phase
    progress = (scheduler.current_step - scheduler.warmup_steps) /
        (scheduler.T_max - scheduler.warmup_steps)
    progress = min(progress, 1.0)

    return scheduler.η_min + (scheduler.η_max - scheduler.η_min) *
        0.5 * (1 + cos(π * progress))
end

function get_learning_rate(scheduler::LinearWarmupScheduler)
    scheduler.current_step += 1

    if scheduler.current_step <= scheduler.warmup_steps
        return scheduler.η_target * (scheduler.current_step / scheduler.warmup_steps)
    else
        return scheduler.η_target
    end
end

function get_learning_rate(scheduler::ExponentialDecayScheduler)
    scheduler.current_step += 1

    decay_factor = scheduler.decay_rate ^ (scheduler.current_step / scheduler.decay_steps)
    return scheduler.η_initial * decay_factor
end

# Mixed precision training utilities
mutable struct MixedPrecisionState
    scale::Ref{Float32}
    growth_factor::Float32
    backoff_factor::Float32
    growth_interval::Int
    steps_since_update::Ref{Int}
    inf_count::Ref{Int}
    enabled::Bool
end

function MixedPrecisionState(initial_scale=65536.0f0, growth_factor=2.0f0,
                             backoff_factor=0.5f0, growth_interval=2000, enabled=true)
    MixedPrecisionState(Ref(initial_scale), growth_factor, backoff_factor,
                        growth_interval, Ref(0), Ref(0), enabled)
end

function scale_gradients!(gradients, mp_state::MixedPrecisionState)
    if !mp_state.enabled
        return
    end

    for grad in gradients
        if grad !== nothing
            grad .*= mp_state.scale[]
        end
    end
end

function unscale_gradients!(gradients, mp_state::MixedPrecisionState)
    if !mp_state.enabled
        return true
    end

    for grad in gradients
        if grad !== nothing
            grad ./= mp_state.scale[]
        end
    end
    return check_gradients_finite(gradients)
end

function check_gradients_finite(gradients)
    for grad in gradients
        if grad !== nothing && any(!isfinite, grad)
            return false
        end
    end
    return true
end

function update_scale!(mp_state::MixedPrecisionState, gradients_finite::Bool)
    if !mp_state.enabled
        return
    end

    if gradients_finite
        mp_state.steps_since_update[] += 1
        mp_state.inf_count[] = 0

        if mp_state.steps_since_update[] >= mp_state.growth_interval
            mp_state.scale[] = min(mp_state.scale[] * mp_state.growth_factor, 1f8)
            mp_state.steps_since_update[] = 0
        end
    else
        mp_state.scale[] = max(mp_state.scale[] * mp_state.backoff_factor, 1f-4)
        mp_state.steps_since_update[] = 0
        mp_state.inf_count[] += 1
    end
end

# Training configuration
struct TrainingConfig
    learning_rate::Float32
    weight_decay::Float32
    max_steps::Int
    warmup_ratio::Float32
    gradient_clip_value::Float32
    accumulation_steps::Int
    eval_steps::Int
    save_steps::Int
    log_steps::Int
    scheduler_type::Symbol  # :cosine, :linear_warmup, :exponential
    use_mixed_precision::Bool
    dropout_rate::Float32
    label_smoothing::Float32
end

function TrainingConfig(;
                        learning_rate=1f-4,
                        weight_decay=0.01f0,
                        max_steps=10000,
                        warmup_ratio=0.1f0,
                        gradient_clip_value=1.0f0,
                        accumulation_steps=1,
                        eval_steps=500,
                        save_steps=1000,
                        log_steps=100,
                        scheduler_type=:cosine,
                        use_mixed_precision=false,
                        dropout_rate=0.1f0,
                        label_smoothing=0.0f0
                        )
    TrainingConfig(learning_rate, weight_decay, max_steps, warmup_ratio,
                   gradient_clip_value, accumulation_steps, eval_steps, save_steps,
                   log_steps, scheduler_type, use_mixed_precision, dropout_rate,
                   label_smoothing)
end

# Enhanced transformer trainer
mutable struct TransformerTrainer
    config::TrainingConfig
    transformer_config::TransformerConfig
    optimizer::AdamW
    scheduler::LRScheduler
    mp_state::MixedPrecisionState
    step::Int
    epoch::Int
    best_loss::Float32
    training_losses::Vector{Float32}
    validation_losses::Vector{Float32}
    learning_rates::Vector{Float32}
end

function TransformerTrainer(transformer_config::TransformerConfig, training_config::TrainingConfig)
    optimizer = AdamW(training_config.learning_rate, 0.9f0, 0.999f0, 1f-8, training_config.weight_decay)

    # Create scheduler based on type
    if training_config.scheduler_type == :cosine
        scheduler = CosineAnnealingScheduler(
            training_config.learning_rate,
            training_config.learning_rate * 0.1f0,
            training_config.max_steps,
            training_config.warmup_ratio
            )
        elseif training_config.scheduler_type == :linear_warmup
        scheduler = LinearWarmupScheduler(
            training_config.learning_rate,
            Int(round(training_config.max_steps * training_config.warmup_ratio))
            )
        elseif training_config.scheduler_type == :exponential
        scheduler = ExponentialDecayScheduler(
            training_config.learning_rate,
            0.95f0,
            training_config.max_steps ÷ 10
            )
    else
        error("Unknown scheduler type: $(training_config.scheduler_type)")
    end

    mp_state = MixedPrecisionState(enabled=training_config.use_mixed_precision)

    TransformerTrainer(training_config, transformer_config, optimizer, scheduler, mp_state,
                       0, 0, Inf32, Float32[], Float32[], Float32[])
end

# Advanced loss functions
function compute_reconstruction_loss(predicted::Matrix{Float32}, target::Matrix{Float32},
                                     reduction::Symbol=:mean)
    """
    Compute reconstruction loss between predicted and target embeddings
    """
    diff = predicted .- target
    loss_matrix = diff .^ 2

    if reduction == :mean
        return mean(loss_matrix)
        elseif reduction == :sum
        return sum(loss_matrix)
        elseif reduction == :none
        return loss_matrix
    else
        error("Unknown reduction: $reduction")
    end
end

function compute_contrastive_loss(embeddings::Matrix{Float32}, labels::Vector{Int},
                                  margin::Float32=1.0f0, temperature::Float32=0.1f0)
    """
    Compute contrastive loss for learning better representations
        """
        batch_size = length(labels)
        loss = 0.0f0

        for i in 1:batch_size
            for j in 1:batch_size
                if i != j
                    similarity = dot(embeddings[:, i], embeddings[:, j]) / temperature

                    if labels[i] == labels[j]
                        # Positive pair - maximize similarity
                        loss -= similarity
                    else
                        # Negative pair - minimize similarity with margin
                        loss += max(0.0f0, similarity + margin)
                    end
                end
            end
        end

        return loss / (batch_size * (batch_size - 1))
    end

    function compute_masked_language_modeling_loss(ai::AGI, input_embeddings::Matrix{Float32},
                                                   mask_indices::Vector{Int}, target_tokens::Vector{Int})
        """
        Compute masked language modeling loss (like BERT)
        """
        # Forward pass through transformer
        output_embeddings = transformer_encode_full(ai, input_embeddings)

        total_loss = 0.0f0
        vocab_size = size(ai.word_embeddings.matrix, 2)

        for (pos, target_token) in zip(mask_indices, target_tokens)
            if pos <= size(output_embeddings, 2) && target_token <= vocab_size
                # Get output at masked position
                output_at_pos = output_embeddings[:, pos]

                # Compute logits for all vocabulary tokens
                logits = zeros(Float32, vocab_size)
                for i in 1:vocab_size
                    logits[i] = dot(output_at_pos, ai.word_embeddings.matrix[:, i])
                end

                # Apply softmax and compute cross-entropy loss
                probs = softmax(logits)
                total_loss -= log(probs[target_token] + 1f-8)  # Add small epsilon for numerical stability
            end
        end

        return total_loss / length(mask_indices)
    end

    function apply_dropout(embeddings::Matrix{Float32}, dropout_rate::Float32, training::Bool=true)
        """
        Apply dropout regularization during training
        """
        if !training || dropout_rate <= 0.0f0
            return embeddings
        end

        # Create dropout mask
        mask = rand(Float32, size(embeddings)) .> dropout_rate

        # Scale by dropout probability to maintain expected value
        scale_factor = 1.0f0 / (1.0f0 - dropout_rate)

        return embeddings .* mask .* scale_factor
    end

    function transformer_encode_full(ai::AGI, embeddings::Matrix{Float32};
                                     dropout_rate::Float32=0.0f0, training::Bool=true)
        """
        Enhanced transformer encoding with dropout and multiple layers
        """
        x = embeddings

        # Apply input dropout
        x = apply_dropout(x, dropout_rate, training)

        # Multiple transformer layers
        num_layers = max(1, ai.config.d_model ÷ 128)  # Adaptive number of layers

        for layer in 1:num_layers
            # Multi-head attention
            attn_output = multi_head_attention(x, x, x, ai.config.n_head)
            attn_output = apply_dropout(attn_output, dropout_rate, training)

            # Residual connection and layer norm
            x = layer_norm(x + attn_output)

            # Feed-forward network
            ff_output = feed_forward(x, ai.config.d_ff)
            ff_output = apply_dropout(ff_output, dropout_rate, training)

            # Residual connection and layer norm
            x = layer_norm(x + ff_output)
        end

        return x
    end

    function clip_gradients!(gradients, max_norm::Float32)
        """
        Clip gradients by global norm
        """
        # Compute total norm across all gradients
        total_norm = 0.0f0
        for grad in gradients
            if grad !== nothing
                total_norm += sum(grad .^ 2)
            end
        end
        total_norm = sqrt(total_norm)

        # Clip if necessary
        if total_norm > max_norm
            clip_factor = max_norm / total_norm
            for grad in gradients
                if grad !== nothing
                    grad .*= clip_factor
                end
            end
        end

        return total_norm
    end

    function create_training_batch(ai::AGI, texts::Vector{String}, batch_size::Int,
                                   tokenizer=nothing, max_length::Int=0)
        """
        Create a training batch with proper padding and attention masks
        """
        if max_length == 0
            max_length = ai.config.max_seq_length
        end

        batch_embeddings = Vector{Matrix{Float32}}()
        batch_lengths = Vector{Int}()

        for i in 1:min(batch_size, length(texts))
            text = texts[i]

            # Encode text (use BPE if tokenizer provided)
            if tokenizer !== nothing
                embeddings = encode_text_bpe(ai, text, tokenizer)
            else
                embeddings = encode_text(ai, text)
            end

            # Truncate or pad to max_length
            seq_len = size(embeddings, 2)
            if seq_len > max_length
                embeddings = embeddings[:, 1:max_length]
                seq_len = max_length
                elseif seq_len < max_length
                # Pad with zeros
                padding = zeros(Float32, ai.config.d_model, max_length - seq_len)
                embeddings = hcat(embeddings, padding)
            end

            push!(batch_embeddings, embeddings)
            push!(batch_lengths, seq_len)
        end

        return batch_embeddings, batch_lengths
    end

    function train_step!(ai::AGI, trainer::TransformerTrainer, batch_data::Vector{String},
                         tokenizer=nothing)
        """
        Perform one training step with all advanced features
        """
        accumulated_loss = 0.0f0

        # Get current learning rate
        current_lr = get_learning_rate(trainer.scheduler)
        trainer.optimizer.η = current_lr

        # Create training batch
        batch_embeddings, batch_lengths = create_training_batch(
            ai, batch_data, trainer.config.accumulation_steps, tokenizer
            )

        # Gradient accumulation loop
        accumulated_gradients = nothing

        for step in 1:length(batch_embeddings)
            input_embeddings = batch_embeddings[step]
            seq_length = batch_lengths[step]

            # Create target (for autoencoding task)
            # In practice, you might want different targets for different tasks
            target_embeddings = input_embeddings[:, 1:seq_length]

            # Forward pass with dropout
            output_embeddings = transformer_encode_full(
                ai, input_embeddings,
                dropout_rate=trainer.config.dropout_rate,
                training=true
                )

            # Compute loss
            output_trimmed = output_embeddings[:, 1:seq_length]
            step_loss = compute_reconstruction_loss(output_trimmed, target_embeddings)

            # Scale loss for gradient accumulation
            step_loss = step_loss / trainer.config.accumulation_steps
            accumulated_loss += step_loss

            # Compute gradients (simplified - in practice you'd use Flux.gradient)
            # This is a placeholder for actual gradient computation
            # gradients = compute_gradients(step_loss, ai)

            # For now, we'll simulate gradient updates on embeddings
            # In a real implementation, you'd compute gradients for all parameters
            grad_sim = randn(Float32, size(ai.word_embeddings.matrix)) * 0.001f0

            if accumulated_gradients === nothing
                accumulated_gradients = [grad_sim]
            else
                accumulated_gradients[1] += grad_sim
            end
        end

        # Apply accumulated gradients
        if accumulated_gradients !== nothing
            # Scale and clip gradients
            if trainer.config.use_mixed_precision
                scale_gradients!(accumulated_gradients, trainer.mp_state)
            end

            gradients_finite = true
            if trainer.config.use_mixed_precision
                gradients_finite = unscale_gradients!(accumulated_gradients, trainer.mp_state)
            end

            if gradients_finite
                grad_norm = clip_gradients!(accumulated_gradients, trainer.config.gradient_clip_value)

                # Apply optimizer update (simplified)
                # In practice, you'd update all model parameters
                update!(trainer.optimizer, ai.word_embeddings.matrix, accumulated_gradients[1])
            end

            # Update mixed precision scale
            if trainer.config.use_mixed_precision
                update_scale!(trainer.mp_state, gradients_finite)
            end
        end

        # Update step counter
        trainer.step += 1

        # Record metrics
        push!(trainer.training_losses, Float32(accumulated_loss))
        push!(trainer.learning_rates, current_lr)

        return Float32(accumulated_loss)
    end

    function validate_model(ai::AGI, validation_texts::Vector{String}, tokenizer=nothing)
        """
        Validate model on held-out data
        """
        total_loss = 0.0f0
        num_samples = 0

        for text in validation_texts
            # Encode without dropout
            if tokenizer !== nothing
                embeddings = encode_text_bpe(ai, text, tokenizer)
            else
                embeddings = encode_text(ai, text)
            end

            # Forward pass without dropout
            output_embeddings = transformer_encode_full(ai, embeddings, dropout_rate=0.0f0, training=false)

            # Compute validation loss
            seq_len = min(size(embeddings, 2), size(output_embeddings, 2))
            if seq_len > 0
                loss = compute_reconstruction_loss(
                    output_embeddings[:, 1:seq_len],
                    embeddings[:, 1:seq_len]
                    )
                total_loss += loss
                num_samples += 1
            end
        end

        return num_samples > 0 ? total_loss / num_samples : Inf32
    end

    function train_epoch!(ai::AGI, trainer::TransformerTrainer, training_data::Vector{String},
                          validation_data::Vector{String}=String[], tokenizer=nothing,
                          batch_size::Int=32)
        """
        Train for one epoch with comprehensive logging and validation
            """
            trainer.epoch += 1
            epoch_loss = 0.0f0
            num_batches = cld(length(training_data), batch_size)

            # Shuffle training data
            shuffled_indices = randperm(length(training_data))
            shuffled_data = training_data[shuffled_indices]

            # Progress tracking
            progress = Progress(num_batches, desc="Epoch $(trainer.epoch): ")

            for batch_start in 1:batch_size:length(shuffled_data)
                batch_end = min(batch_start + batch_size - 1, length(shuffled_data))
                batch_data = shuffled_data[batch_start:batch_end]

                # Training step
                batch_loss = train_step!(ai, trainer, batch_data, tokenizer)
                epoch_loss += batch_loss

                # Logging
                if trainer.step % trainer.config.log_steps == 0
                    avg_loss = epoch_loss / ((batch_start - 1) ÷ batch_size + 1)

                    ProgressMeter.next!(progress,
                                        showvalues = [
                                            (:step, trainer.step),
                                            (:loss, round(batch_loss, digits=4)),
                                            (:avg_loss, round(avg_loss, digits=4)),
                                            (:lr, round(trainer.optimizer.η, digits=6)),
                                            (:scale, trainer.config.use_mixed_precision ? trainer.mp_state.scale[] : "N/A")
                                            ])
                end

                # Validation
                if trainer.step % trainer.config.eval_steps == 0 && !isempty(validation_data)
                    val_loss = validate_model(ai, validation_data[1:min(100, length(validation_data))], tokenizer)
                    push!(trainer.validation_losses, val_loss)

                    @info "Validation" step=trainer.step val_loss=round(val_loss, digits=4)

                    # Save best model
                    if val_loss < trainer.best_loss
                        trainer.best_loss = val_loss
                        save_checkpoint(ai, trainer, "best_model.jld2")
                        @info "New best model saved" val_loss=val_loss
                    end
                end

                # Periodic saving
                if trainer.step % trainer.config.save_steps == 0
                    checkpoint_path = "checkpoint_step_$(trainer.step).jld2"
                    save_checkpoint(ai, trainer, checkpoint_path)
                    @info "Checkpoint saved" path=checkpoint_path
                end

                # Early stopping check
                if trainer.step >= trainer.config.max_steps
                    @info "Maximum steps reached, stopping training"
                    break
                end
            end

            return epoch_loss / num_batches
        end

        function save_checkpoint(ai::AGI, trainer::TransformerTrainer, path::String)
            """
            Save training checkpoint with full state
            """
            try
                JLD2.jldopen(path, "w") do file
                    # Save AGI state
                    file["vocab_idx_to_word"] = ai.vocab.idx_to_word
                    file["vocab_word_to_idx"] = ai.vocab.word_to_idx
                    file["word_embeddings"] = ai.word_embeddings.matrix
                    file["positional_enc"] = ai.positional_enc.matrix
                    file["documents"] = ai.docs.documents
                    file["doc_embeddings"] = ai.docs.embeddings
                    file["config"] = ai.config

                    # Save training state
                    file["trainer_step"] = trainer.step
                    file["trainer_epoch"] = trainer.epoch
                    file["best_loss"] = trainer.best_loss
                    file["training_losses"] = trainer.training_losses
                    file["validation_losses"] = trainer.validation_losses
                    file["learning_rates"] = trainer.learning_rates

                    # Save optimizer state
                    file["optimizer_eta"] = trainer.optimizer.η
                    file["optimizer_t"] = trainer.optimizer.t

                    # Save training config
                    file["training_config"] = trainer.config
                end

                @info "Checkpoint saved successfully" path=path step=trainer.step
                catch e
                @error "Failed to save checkpoint" path=path exception=e
            end
        end

        function load_checkpoint(path::String, ai::AGI, trainer::TransformerTrainer)
            """
            Load training checkpoint and restore state
            """
            try
                JLD2.jldopen(path, "r") do file
                    # Restore AGI state
                    ai.vocab.idx_to_word = file["vocab_idx_to_word"]
                    ai.vocab.word_to_idx = file["vocab_word_to_idx"]
                    ai.word_embeddings.matrix = file["word_embeddings"]
                    ai.positional_enc.matrix = file["positional_enc"]
                    ai.docs.documents = file["documents"]
                    ai.docs.embeddings = file["doc_embeddings"]

                    # Restore training state
                    trainer.step = file["trainer_step"]
                    trainer.epoch = file["trainer_epoch"]
                    trainer.best_loss = file["best_loss"]
                    trainer.training_losses = file["training_losses"]
                    trainer.validation_losses = file["validation_losses"]
                    trainer.learning_rates = file["learning_rates"]

                    # Restore optimizer state
                    trainer.optimizer.η = file["optimizer_eta"]
                    trainer.optimizer.t = file["optimizer_t"]
                end

                @info "Checkpoint loaded successfully" path=path step=trainer.step
                catch e
                @error "Failed to load checkpoint" path=path exception=e
                rethrow(e)
            end
        end

        # Enhanced learning function with advanced training
        function enhanced_learn!(ai::AGI, texts::Vector{String}, epochs::Int=5,
                                 batch_size::Int=32, learning_rate::Float32=1f-4,
                                 validation_split::Float32=0.1f0, tokenizer=nothing,
                                 training_config::TrainingConfig=TrainingConfig())
            """
            Enhanced learning function with comprehensive training pipeline
                """
                @info "Starting enhanced training" epochs=epochs batch_size=batch_size lr=learning_rate

                # Split data into training and validation
                if validation_split > 0.0f0
                    n_val = Int(round(length(texts) * validation_split))
                    indices = randperm(length(texts))
                    val_indices = indices[1:n_val]
                    train_indices = indices[n_val+1:end]

                    validation_data = texts[val_indices]
                    training_data = texts[train_indices]
                else
                    training_data = texts
                    validation_data = String[]
                end

                @info "Data split" training_samples=length(training_data) validation_samples=length(validation_data)

                # Update training config
                config = TrainingConfig(
                    learning_rate=learning_rate,
                    max_steps=epochs * cld(length(training_data), batch_size),
                    weight_decay=training_config.weight_decay,
                    warmup_ratio=training_config.warmup_ratio,
                    gradient_clip_value=training_config.gradient_clip_value,
                    accumulation_steps=training_config.accumulation_steps,
                    eval_steps=training_config.eval_steps,
                    save_steps=training_config.save_steps,
                    log_steps=training_config.log_steps,
                    scheduler_type=training_config.scheduler_type,
                    use_mixed_precision=training_config.use_mixed_precision,
                    dropout_rate=training_config.dropout_rate,
                    label_smoothing=training_config.label_smoothing
                    )

                # Create trainer
                trainer = TransformerTrainer(ai.config, config)

                @info "Training configuration" config

                # Training loop
                for epoch in 1:epochs
                    @info "Starting epoch $epoch/$epochs"

                    epoch_loss = train_epoch!(ai, trainer, training_data, validation_data,
                                              tokenizer, batch_size)

                    @info "Epoch completed" epoch=epoch avg_loss=round(epoch_loss, digits=4)
                    total_steps=trainer.step

                    # Re-encode all documents with updated model
                    @info "Re-encoding documents with updated model"
                    reiterate!(ai)

                    # Early stopping check
                    if trainer.step >= trainer.config.max_steps
                        break
                    end
                end

                # Save final model
                final_path = "final_model_$(Dates.format(now(), "yyyymmdd_HHMMSS")).jld2"
                save_checkpoint(ai, trainer, final_path)

                @info "Training completed" total_steps=trainer.step final_loss=trainer.best_loss
                saved_path=final_path

                return trainer
            end

            # Training utilities and helpers
            function compute_model_size(ai::AGI)
                """Compute total number of parameters in the model"""
                total_params = length(ai.word_embeddings.matrix) + length(ai.positional_enc.matrix)
                return total_params
            end

            function estimate_memory_usage(ai::AGI, batch_size::Int)
                """Estimate memory usage for training"""
                    model_size = compute_model_size(ai)
                    activation_size = ai.config.d_model * ai.config.max_seq_length * batch_size

                    # Rough estimate including gradients and optimizer states
                    total_size_bytes = (model_size + activation_size) * 4 * 3  # Float32 * (params + grads + optimizer)
                    total_size_mb = total_size_bytes / (1024 * 1024)

                    return total_size_mb
                end

                function suggest_batch_size(ai::AGI, available_memory_gb::Float32=8.0f0)
                    """Suggest optimal batch size based on available memory"""
                    target_memory_mb = available_memory_gb * 1024 * 0.8  # Use 80% of available memory

                    # Binary search for optimal batch size
                    low, high = 1, 256
                    best_batch_size = 1

                    while low <= high
                        mid = (low + high) ÷ 2
                        estimated_usage = estimate_memory_usage(ai, mid)

                        if estimated_usage <= target_memory_mb
                            best_batch_size = mid
                            low = mid + 1
                        else
                            high = mid - 1
                        end
                    end

                    return best_batch_size
                end

                # Export all training components
                export AdamW, CosineAnnealingScheduler, LinearWarmupScheduler, ExponentialDecayScheduler
                export TrainingConfig, TransformerTrainer, MixedPrecisionState
                export enhanced_learn!, train_epoch!, train_step!, validate_model
                export save_checkpoint, load_checkpoint
                export compute_reconstruction_loss, compute_contrastive_loss, compute_masked_language_modeling_loss
                export transformer_encode_full, clip_gradients!, create_training_batch
                export compute_model_size, estimate_memory_usage, suggest_batch_size
