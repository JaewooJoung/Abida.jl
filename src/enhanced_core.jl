# enhanced_core.jl - Enhanced core functionality with generation and advanced features

using Random
using StatsBase
using Statistics
using LinearAlgebra
using Dates
using JSON3
using ProgressMeter
using Logging

# Enhanced sampling functions for text generation
function sample_token(logits::Vector{Float32}; temperature::Float32=1.0f0, 
                     top_p::Float32=0.9f0, top_k::Int=50, repetition_penalty::Float32=1.0f0,
                     recent_tokens::Vector{Int}=Int[])
    """
    Advanced sampling with temperature, top-p, top-k filtering, and repetition penalty
    """
    if temperature <= 0.0f0
        return argmax(logits)
    end
    
    # Apply repetition penalty
    if repetition_penalty != 1.0f0 && !isempty(recent_tokens)
        for token_id in recent_tokens
            if token_id <= length(logits)
                if logits[token_id] > 0
                    logits[token_id] /= repetition_penalty
                else
                    logits[token_id] *= repetition_penalty
                end
            end
        end
    end
    
    # Apply temperature
    scaled_logits = logits ./ temperature
    
    # Top-k filtering
    if top_k > 0 && top_k < length(scaled_logits)
        top_k_indices = partialsortperm(scaled_logits, 1:top_k, rev=true)
        filtered_logits = fill(-Inf32, length(scaled_logits))
        filtered_logits[top_k_indices] = scaled_logits[top_k_indices]
        scaled_logits = filtered_logits
    end
    
    # Convert to probabilities
    probs = softmax(scaled_logits)
    
    # Top-p (nucleus) filtering
    if top_p < 1.0f0
        sorted_indices = sortperm(probs, rev=true)
        cumsum_probs = cumsum(probs[sorted_indices])
        
        # Find cutoff point
        cutoff_idx = findfirst(x -> x >= top_p, cumsum_probs)
        if cutoff_idx !== nothing
            # Zero out probabilities beyond cutoff
            keep_indices = sorted_indices[1:cutoff_idx]
            filtered_probs = zeros(Float32, length(probs))
            filtered_probs[keep_indices] = probs[keep_indices]
            probs = filtered_probs ./ sum(filtered_probs)  # Renormalize
        end
    end
    
    # Sample from the distribution
    return sample(1:length(probs), Weights(probs))
end

function sample_with_constraints(logits::Vector{Float32}, 
                                forbidden_tokens::Set{Int}=Set{Int}(),
                                required_tokens::Set{Int}=Set{Int}(),
                                bias_tokens::Dict{Int, Float32}=Dict{Int, Float32}();
                                kwargs...)
    """
    Sample with token constraints and biases
    """
    # Apply forbidden tokens
    for token_id in forbidden_tokens
        if token_id <= length(logits)
            logits[token_id] = -Inf32
        end
    end
    
    # Apply required tokens (if any are available)
    if !isempty(required_tokens)
        available_required = [t for t in required_tokens if t <= length(logits)]
        if !isempty(available_required)
            # Zero out all tokens except required ones
            new_logits = fill(-Inf32, length(logits))
            for token_id in available_required
                new_logits[token_id] = logits[token_id]
            end
            logits = new_logits
        end
    end
    
    # Apply token biases
    for (token_id, bias) in bias_tokens
        if token_id <= length(logits)
            logits[token_id] += bias
        end
    end
    
    return sample_token(logits; kwargs...)
end

# Enhanced text generation capabilities
struct GenerationConfig
    max_tokens::Int
    temperature::Float32
    top_p::Float32
    top_k::Int
    repetition_penalty::Float32
    repetition_window::Int
    stop_tokens::Vector{String}
    stop_sequences::Vector{String}
    seed::Union{Int, Nothing}
    early_stopping::Bool
    min_length::Int
    length_penalty::Float32
end

function GenerationConfig(;
    max_tokens=100,
    temperature=0.7f0,
    top_p=0.9f0,
    top_k=50,
    repetition_penalty=1.0f0,
    repetition_window=20,
    stop_tokens=String[],
    stop_sequences=String[],
    seed=nothing,
    early_stopping=true,
    min_length=1,
    length_penalty=1.0f0
)
    GenerationConfig(max_tokens, temperature, top_p, top_k, repetition_penalty,
                    repetition_window, stop_tokens, stop_sequences, seed,
                    early_stopping, min_length, length_penalty)
end

function generate_text(ai::AGI, prompt::String, config::GenerationConfig=GenerationConfig();
                      tokenizer=nothing, verbose::Bool=false)
    """
    Generate text using the trained model with advanced sampling and constraints
    """
    if isempty(ai.vocab.word_to_idx)
        return "No vocabulary available for generation."
    end
    
    # Set random seed if specified
    if config.seed !== nothing
        Random.seed!(config.seed)
    end
    
    # Tokenize prompt
    if tokenizer !== nothing
        tokens = tokenize(tokenizer, prompt)
        token_ids = tokens_to_ids(tokenizer, tokens)
    else
        words = split(lowercase(prompt))
        token_ids = [get(ai.vocab.word_to_idx, word, 1) for word in words]
        tokens = words
    end
    
    generated_token_ids = copy(token_ids)
    generated_tokens = copy(tokens)
    
    if verbose
        @info "Starting generation" prompt_length=length(tokens) config=config
    end
    
    # Generation loop
    for step in 1:config.max_tokens
        # Take last context window
        context_length = min(length(generated_token_ids), ai.config.max_seq_length)
        context_ids = generated_token_ids[end-context_length+1:end]
        
        # Encode context
        context_embeddings = zeros(Float32, ai.config.d_model, context_length)
        
        for (i, token_id) in enumerate(context_ids)
            if token_id <= size(ai.word_embeddings.matrix, 2)
                context_embeddings[:, i] = ai.word_embeddings.matrix[:, token_id]
            else
                context_embeddings[:, i] = randn(Float32, ai.config.d_model) * 0.1f0
            end
            
            # Add positional encoding
            if i <= size(ai.positional_enc.matrix, 2)
                context_embeddings[:, i] += ai.positional_enc.matrix[:, i]
            end
        end
        
        # Forward pass through transformer
        output_embeddings = transformer_encode_full(ai, context_embeddings, dropout_rate=0.0f0, training=false)
        
        # Get last token's output for next token prediction
        last_output = output_embeddings[:, end]
        
        # Compute logits (similarity to all word embeddings)
        vocab_size = size(ai.word_embeddings.matrix, 2)
        logits = zeros(Float32, vocab_size)
        
        for i in 1:vocab_size
            if i <= length(ai.vocab.idx_to_word)
                logits[i] = dot(last_output, ai.word_embeddings.matrix[:, i])
            end
        end
        
        # Apply length penalty
        if config.length_penalty != 1.0f0
            length_factor = (length(generated_tokens) / config.max_tokens) ^ config.length_penalty
            logits .*= length_factor
        end
        
        # Get recent tokens for repetition penalty
        recent_window = min(config.repetition_window, length(generated_token_ids))
        recent_tokens = generated_token_ids[end-recent_window+1:end]
        
        # Sample next token
        next_token_id = sample_token(logits, 
                                   temperature=config.temperature,
                                   top_p=config.top_p, 
                                   top_k=config.top_k,
                                   repetition_penalty=config.repetition_penalty,
                                   recent_tokens=recent_tokens)
        
        # Convert token ID to token
        if next_token_id <= length(ai.vocab.idx_to_word)
            next_token = ai.vocab.idx_to_word[next_token_id]
        else
            if verbose
                @warn "Invalid token ID generated" token_id=next_token_id vocab_size=length(ai.vocab.idx_to_word)
            end
            break
        end
        
        # Check for stop tokens
        if next_token in config.stop_tokens
            if verbose
                @info "Stop token encountered" token=next_token step=step
            end
            break
        end
        
        # Add to generated sequence
        push!(generated_token_ids, next_token_id)
        push!(generated_tokens, next_token)
        
        # Check for stop sequences
        recent_text = join(generated_tokens[max(1, end-10):end], " ")
        stop_found = false
        for stop_seq in config.stop_sequences
            if occursin(stop_seq, recent_text)
                if verbose
                    @info "Stop sequence encountered" sequence=stop_seq step=step
                end
                stop_found = true
                break
            end
        end
        
        if stop_found && config.early_stopping
            break
        end
        
        # Check minimum length
        if step >= config.min_length && config.early_stopping
            # Can stop early if we've reached minimum length
        end
    end
    
    # Return only the generated part
    generated_part = generated_tokens[length(tokens)+1:end]
    result = join(generated_part, " ")
    
    if verbose
        @info "Generation completed" generated_tokens=length(generated_part) final_length=length(result)
    end
    
    return result
end

function generate_with_prompt_template(ai::AGI, template::String, variables::Dict{String, String},
                                     config::GenerationConfig=GenerationConfig(); kwargs...)
    """
    Generate text using a prompt template with variable substitution
    """
    # Substitute variables in template
    prompt = template
    for (key, value) in variables
        placeholder = "{$key}"
        prompt = replace(prompt, placeholder => value)
    end
    
    return generate_text(ai, prompt, config; kwargs...)
end

function generate_multiple_candidates(ai::AGI, prompt::String, num_candidates::Int=3,
                                    config::GenerationConfig=GenerationConfig(); kwargs...)
    """
    Generate multiple candidate responses and optionally rank them
    """
    candidates = String[]
    
    for i in 1:num_candidates
        # Use different random seeds for diversity
        candidate_config = GenerationConfig(
            config.max_tokens, config.temperature, config.top_p, config.top_k,
            config.repetition_penalty, config.repetition_window, config.stop_tokens,
            config.stop_sequences, config.seed === nothing ? nothing : config.seed + i,
            config.early_stopping, config.min_length, config.length_penalty
        )
        
        candidate = generate_text(ai, prompt, candidate_config; kwargs...)
        push!(candidates, candidate)
    end
    
    return candidates
end

# Enhanced question answering with generation
function generate_answer(ai::AGI, vector_store::VectorStore, question::String; 
                        max_answer_length::Int=150, use_rag::Bool=true,
                        config::GenerationConfig=GenerationConfig(),
                        answer_style::String="informative")
    """
    Generate a comprehensive answer by combining retrieval and generation
    """
    if use_rag && !isempty(vector_store.chunks)
        # Get relevant context
        relevant_chunks = hybrid_search(ai, vector_store, question, 3)
        
        if !isempty(relevant_chunks)
            # Create context-aware prompt based on answer style
            context = join([chunk.content for chunk in relevant_chunks], "\n\n")
            
            prompt = if answer_style == "conversational"
                "Based on this information:\n$context\n\nUser asks: $question\n\nI'll explain this in a friendly way:"
            elseif answer_style == "technical"
                "Technical documentation:\n$context\n\nQuestion: $question\n\nTechnical answer:"
            elseif answer_style == "summary"
                "Source material:\n$context\n\nQuestion: $question\n\nSummary:"
            else  # informative
                "Context:\n$context\n\nQuestion: $question\n\nAnswer:"
            end
            
            # Generate answer with appropriate config
            generation_config = GenerationConfig(
                max_answer_length,
                config.temperature,
                config.top_p,
                config.top_k,
                config.repetition_penalty,
                config.repetition_window,
                vcat(config.stop_tokens, ["Question:", "Context:", "User asks:"]),
                config.stop_sequences,
                config.seed,
                config.early_stopping,
                config.min_length,
                config.length_penalty
            )
            
            generated = generate_text(ai, prompt, generation_config)
            
            if !isempty(strip(generated))
                return (strip(generated), 0.8f0, "generated_with_context")
            end
        end
    end
    
    # Fallback to simple generation
    prompt = if answer_style == "conversational"
        "User asks: $question\n\nI'll explain this clearly:"
    else
        "Question: $question\nAnswer:"
    end
    
    fallback_config = GenerationConfig(
        max_answer_length,
        0.5f0,  # Lower temperature for more focused answers
        config.top_p,
        config.top_k,
        config.repetition_penalty,
        config.repetition_window,
        vcat(config.stop_tokens, ["Question:", "User asks:"]),
        config.stop_sequences,
        config.seed,
        config.early_stopping,
        config.min_length,
        config.length_penalty
    )
    
    generated = generate_text(ai, prompt, fallback_config)
    
    return (strip(generated), 0.5f0, "generated")
end

# Conversation management
mutable struct ConversationState
    history::Vector{Tuple{String, String, DateTime}}  # (user, assistant, timestamp)
    context_length::Int
    personality::String
    max_history::Int
    conversation_id::String
    metadata::Dict{String, Any}
    memory_keywords::Set{String}
    user_preferences::Dict{String, Any}
end

function ConversationState(; 
    context_length::Int=512, 
    personality::String="helpful and informative",
    max_history::Int=10,
    conversation_id::String=string(uuid4())[1:8],
    metadata::Dict{String, Any}=Dict{String, Any}()
)
    ConversationState(
        Tuple{String, String, DateTime}[], 
        context_length, 
        personality, 
        max_history,
        conversation_id,
        metadata,
        Set{String}(),
        Dict{String, Any}()
    )
end

function add_exchange!(conv::ConversationState, user_msg::String, assistant_msg::String)
    push!(conv.history, (user_msg, assistant_msg, now()))
    
    # Extract keywords from user message
    user_words = Set(split(lowercase(user_msg)))
    important_words = filter(w -> length(w) > 3 && !in(w, ["this", "that", "with", "from", "they", "have", "will"]), user_words)
    union!(conv.memory_keywords, important_words)
    
    # Trim history if too long
    while length(conv.history) > conv.max_history
        popfirst!(conv.history)
    end
end

function build_conversation_context(conv::ConversationState)
    if isempty(conv.history)
        base_prompt = "You are a $(conv.personality) AI assistant."
        if !isempty(conv.memory_keywords)
            base_prompt *= " You remember discussing: $(join(collect(conv.memory_keywords)[1:min(5, length(conv.memory_keywords))], ", "))."
        end
        return base_prompt
    end
    
    context_parts = ["You are a $(conv.personality) AI assistant."]
    
    if !isempty(conv.user_preferences)
        prefs = [string(k) * ": " * string(v) for (k,v) in conv.user_preferences]
        push!(context_parts, "User preferences: " * join(prefs, ", "))
    end
    
    push!(context_parts, "\nConversation history:")
    
    for (user_msg, assistant_msg, timestamp) in conv.history
        push!(context_parts, "User: $user_msg")
        push!(context_parts, "Assistant: $assistant_msg")
    end
    
    return join(context_parts, "\n")
end

function conversational_answer(ai::AGI, vector_store::VectorStore, 
                             conv::ConversationState, user_input::String;
                             config::GenerationConfig=GenerationConfig())
    """
    Generate contextual responses considering conversation history
    """
    # Build conversation context
    context = build_conversation_context(conv)
    
    # Create full prompt
    full_prompt = "$context\nUser: $user_input\nAssistant:"
    
    # Generate response with conversation-appropriate config
    conv_config = GenerationConfig(
        config.max_tokens,
        0.7f0,  # Slightly higher temperature for more natural conversation
        config.top_p,
        config.top_k,
        1.1f0,  # Slight repetition penalty for more diverse responses
        config.repetition_window,
        vcat(config.stop_tokens, ["User:", "Assistant:"]),
        config.stop_sequences,
        config.seed,
        config.early_stopping,
        config.min_length,
        config.length_penalty
    )
    
    response = generate_text(ai, full_prompt, conv_config)
    
    # Clean up response
    response = strip(response)
    
    # Add to conversation history
    add_exchange!(conv, user_input, response)
    
    return response
end

function update_user_preferences!(conv::ConversationState, preferences::Dict{String, Any})
    """
    Update user preferences for personalized responses
    """
    merge!(conv.user_preferences, preferences)
end

function get_conversation_summary(conv::ConversationState)
    """
    Get a summary of the conversation
    """
    if isempty(conv.history)
        return "No conversation history"
    end
    
    total_exchanges = length(conv.history)
    first_exchange = conv.history[1][3]
    last_exchange = conv.history[end][3]
    duration = last_exchange - first_exchange
    
    key_topics = join(collect(conv.memory_keywords)[1:min(5, length(conv.memory_keywords))], ", ")
    
    return """
    Conversation Summary:
    - ID: $(conv.conversation_id)
    - Exchanges: $total_exchanges
    - Duration: $(Dates.format(duration, "HH:MM:SS"))
    - Key topics: $key_topics
    - Personality: $(conv.personality)
    """
end

# Advanced learning with feedback integration
function learn_from_feedback!(ai::AGI, question::String, 
                             provided_answer::String, feedback::String, 
                             is_positive::Bool, feedback_type::String="general")
    """
    Learn from user feedback to improve future responses
    """
    try
        # Store feedback in database with additional metadata
        feedback_metadata = Dict{String, Any}(
            "feedback_type" => feedback_type,
            "answer_length" => length(provided_answer),
            "question_length" => length(question),
            "timestamp" => now(),
            "confidence_before" => 0.5  # Would be actual confidence if available
        )
        
        DBInterface.execute(ai.conn, """
            INSERT INTO feedback (id, question, feedback, is_positive, metadata) 
            VALUES (nextval('feedback_id_seq'), ?, ?, ?, ?)
        """, (question, feedback, is_positive, JSON3.write(feedback_metadata)))
        
        # If feedback is positive, reinforce the knowledge
        if is_positive
            if feedback_type == "factual"
                reinforcement_text = "FACT: Q: $question A: $provided_answer"
            elseif feedback_type == "style"
                reinforcement_text = "GOOD_STYLE: Q: $question A: $provided_answer"
            else
                reinforcement_text = "POSITIVE: Q: $question A: $provided_answer"
            end
            learn!(ai, reinforcement_text)
        else
            # If feedback is negative, learn the corrected version
            if !isempty(feedback)
                if feedback_type == "factual"
                    corrected_text = "CORRECTED_FACT: Q: $question A: $feedback"
                elseif feedback_type == "style"
                    corrected_text = "BETTER_STYLE: Q: $question A: $feedback"
                else
                    corrected_text = "CORRECTED: Q: $question A: $feedback"
                end
                learn!(ai, corrected_text)
            end
        end
        
        @info "Feedback processed" positive=is_positive type=feedback_type question=question[1:min(50, length(question))]
        
    catch e
        @warn "Failed to process feedback" exception=e
    end
end

function analyze_feedback_patterns(ai::AGI)
    """
    Analyze feedback patterns to identify improvement areas
    """
    try
        result = DBInterface.execute(ai.conn, """
            SELECT 
                is_positive,
                COUNT(*) as count,
                AVG(LENGTH(question)) as avg_question_length,
                AVG(LENGTH(feedback)) as avg_feedback_length
            FROM feedback 
            GROUP BY is_positive
        """)
        
        patterns = Dict{String, Any}()
        for row in result
            key = row.is_positive ? "positive" : "negative"
            patterns[key] = Dict{String, Any}(
                "count" => row.count,
                "avg_question_length" => row.avg_question_length,
                "avg_feedback_length" => row.avg_feedback_length
            )
        end
        
        return patterns
    catch e
        @warn "Failed to analyze feedback patterns" exception=e
        return Dict{String, Any}()
    end
end

# Model evaluation and metrics
struct EvaluationResult
    total_questions::Int
    correct_answers::Int
    accuracy::Float64
    average_confidence::Float64
    average_response_time::Float64
    detailed_results::Vector{Dict{String, Any}}
    response_times::Vector{Float64}
    confidence_scores::Vector{Float64}
    evaluation_timestamp::DateTime
end

function evaluate_model(ai::AGI, test_questions::Vector{String}, 
                       expected_answers::Vector{String};
                       vector_store=nothing,
                       evaluation_method::String="similarity",
                       similarity_threshold::Float64=0.7)
    """
    Evaluate model performance on a test set with multiple evaluation methods
    """
    if length(test_questions) != length(expected_answers)
        error("Number of questions and answers must match")
    end
    
    @info "Starting model evaluation" questions=length(test_questions) method=evaluation_method
    
    detailed_results = Dict{String, Any}[]
    response_times = Float64[]
    confidence_scores = Float64[]
    correct_count = 0
    total_confidence = 0.0
    
    progress = Progress(length(test_questions), desc="Evaluating: ")
    
    for (i, (question, expected)) in enumerate(zip(test_questions, expected_answers))
        start_time = time()
        
        # Get model answer based on method
        if evaluation_method == "generation" && vector_store !== nothing
            answer, confidence, source = generate_answer(ai, vector_store, question)
        elseif vector_store !== nothing
            answer, confidence, source = comprehensive_answer(ai, vector_store, question)
        else
            answer, confidence, source = answer(ai, question)
        end
        
        response_time = time() - start_time
        push!(response_times, response_time)
        push!(confidence_scores, confidence)
        
        # Evaluate correctness using different methods
        is_correct = if evaluation_method == "exact"
            lowercase(strip(answer)) == lowercase(strip(expected))
        elseif evaluation_method == "contains"
            occursin(lowercase(expected), lowercase(answer)) || 
            occursin(lowercase(answer), lowercase(expected))
        elseif evaluation_method == "similarity"
            # Use embedding similarity
            answer_emb = normalize(vec(mean(encode_text(ai, answer), dims=2)))
            expected_emb = normalize(vec(mean(encode_text(ai, expected), dims=2)))
            similarity = dot(answer_emb, expected_emb)
            similarity >= similarity_threshold
        else  # semantic
            # More sophisticated semantic evaluation
            semantic_similarity = evaluate_semantic_similarity(ai, answer, expected)
            semantic_similarity >= similarity_threshold
        end
        
        if is_correct
            correct_count += 1
        end
        
        total_confidence += confidence
        
        # Store detailed result
        push!(detailed_results, Dict{String, Any}(
            "question" => question,
            "expected" => expected,
            "actual" => answer,
            "correct" => is_correct,
            "confidence" => confidence,
            "response_time" => response_time,
            "source" => source,
            "question_length" => length(question),
            "answer_length" => length(answer)
        ))
        
        ProgressMeter.next!(progress, showvalues = [
            (:progress, "$i/$(length(test_questions))"),
            (:accuracy, "$(round(correct_count/i*100, digits=1))%"),
            (:avg_time, "$(round(mean(response_times)*1000, digits=1))ms")
        ])
    end
    
    accuracy = correct_count / length(test_questions)
    avg_confidence = total_confidence / length(test_questions)
    avg_response_time = mean(response_times)
    
    @info "Evaluation completed" accuracy=round(accuracy*100, digits=1) avg_confidence=round(avg_confidence, digits=3)
    
    return EvaluationResult(
        length(test_questions),
        correct_count,
        accuracy,
        avg_confidence,
        avg_response_time,
        detailed_results,
        response_times,
        confidence_scores,
        now()
    )
end

function evaluate_semantic_similarity(ai::AGI, answer::String, expected::String)
    """
    Evaluate semantic similarity between answer and expected response
    """
    # Simple semantic similarity using embeddings
    answer_words = split(lowercase(answer))
    expected_words = split(lowercase(expected))
    
    # Word overlap similarity
    overlap = length(intersect(Set(answer_words), Set(expected_words)))
    total_words = length(union(Set(answer_words), Set(expected_words)))
    word_similarity = overlap / max(total_words, 1)
    
    # Embedding similarity
    answer_emb = normalize(vec(mean(encode_text(ai, answer), dims=2)))
    expected_emb = normalize(vec(mean(encode_text(ai, expected), dims=2)))
    embedding_similarity = dot(answer_emb, expected_emb)
    
    # Combined similarity (weighted average)
    return 0.3 * word_similarity + 0.7 * embedding_similarity
end

function generate_evaluation_report(result::EvaluationResult, output_path::String="")
    """
    Generate a comprehensive evaluation report
    """
    report = """
    # Model Evaluation Report
    
    **Generated:** $(Dates.format(result.evaluation_timestamp, "yyyy-mm-dd HH:MM:SS"))
    
    ## Overall Performance
    - **Total Questions:** $(result.total_questions)
    - **Correct Answers:** $(result.correct_answers)
    - **Accuracy:** $(round(result.accuracy * 100, digits=2))%
    - **Average Confidence:** $(round(result.average_confidence, digits=3))
    - **Average Response Time:** $(round(result.average_response_time * 1000, digits=1)) ms
    
    ## Performance Distribution
    - **Response Time P50:** $(round(quantile(result.response_times, 0.5) * 1000, digits=1)) ms
    - **Response Time P95:** $(round(quantile(result.response_times, 0.95) * 1000, digits=1)) ms
    - **Confidence P50:** $(round(quantile(result.confidence_scores, 0.5), digits=3))
    - **Confidence P95:** $(round(quantile(result.confidence_scores, 0.95), digits=3))
    
    ## Detailed Results
    """
    
    # Add detailed results
    for (i, result_detail) in enumerate(result.detailed_results)
        status = result_detail["correct"] ? "✅" : "❌"
        report *= """
        
        ### Question $i $status
        **Q:** $(result_detail["question"])
        **Expected:** $(result_detail["expected"])
        **Got:** $(result_detail["actual"])
        **Confidence:** $(round(result_detail["confidence"], digits=2))
        **Time:** $(round(result_detail["response_time"] * 1000, digits=1)) ms
        """
    end
    
    if !isempty(output_path)
        try
            write(output_path, report)
            @info "Evaluation report saved" path=output_path
        catch e
            @warn "Failed to save evaluation report" path=output_path exception=e
        end
    end
    
    return report
end

# Memory management and optimization
function optimize_memory!(ai::AGI, vector_store::VectorStore, 
                         similarity_threshold::Float32=0.95f0,
                         max_documents::Int=0,
                         max_vocabulary::Int=0)
    """
    Comprehensive memory optimization
    """
    @info "Starting comprehensive memory optimization"
    
    initial_docs = length(ai.docs.documents)
    initial_vocab = length(ai.vocab.word_to_idx)
    initial_chunks = length(vector_store.chunks)
    
    # 1. Optimize vector store (remove duplicates)
    optimize_vector_store!(vector_store, similarity_threshold)
    
    # 2. Remove duplicate documents from main store
    unique_docs = Dict{UInt64, Int}()
    documents_to_remove = Int[]
    
    for (i, doc) in enumerate(ai.docs.documents)
        doc_hash = hash(doc)
        if haskey(unique_docs, doc_hash)
            push!(documents_to_remove, i)
        else
            unique_docs[doc_hash] = i
        end
    end
    
    # Remove duplicates (in reverse order to maintain indices)
    for idx in reverse(documents_to_remove)
        deleteat!(ai.docs.documents, idx)
        if idx <= length(ai.docs.embeddings)
            deleteat!(ai.docs.embeddings, idx)
        end
    end
    
    # 3. Trim documents if max_documents is specified
    if max_documents > 0 && length(ai.docs.documents) > max_documents
        # Keep most recent documents
        keep_count = max_documents
        ai.docs.documents = ai.docs.documents[end-keep_count+1:end]
        if length(ai.docs.embeddings) >= keep_count
            ai.docs.embeddings = ai.docs.embeddings[end-keep_count+1:end]
        end
    end
    
    # 4. Optimize vocabulary if needed
    if max_vocabulary > 0 && length(ai.vocab.word_to_idx) > max_vocabulary
        # Keep most frequent words (this would require frequency tracking)
        @warn "Vocabulary trimming not fully implemented - requires frequency tracking"
    end
    
    # 5. Force garbage collection
    GC.gc()
    
    final_docs = length(ai.docs.documents)
    final_vocab = length(ai.vocab.word_to_idx)
    final_chunks = length(vector_store.chunks)
    
    @info "Memory optimization completed" 
          docs_removed=initial_docs-final_docs
          vocab_unchanged=initial_vocab-final_vocab
          chunks_removed=initial_chunks-final_chunks
          final_docs=final_docs
          final_chunks=final_chunks
end

function get_memory_usage_stats(ai::AGI, vector_store::VectorStore)
    """
    Get detailed memory usage statistics
    """
    # Calculate approximate memory usage
    embeddings_size = sizeof(ai.word_embeddings.matrix)
    positional_enc_size = sizeof(ai.positional_enc.matrix)
    docs_size = sum(sizeof(doc) for doc in ai.docs.documents)
    doc_embeddings_size = sum(sizeof(emb) for emb in ai.docs.embeddings)
    vector_store_size = sizeof(vector_store.embeddings_matrix) + 
                       sum(sizeof(chunk.content) + sizeof(chunk.embedding) for chunk in vector_store.chunks)
    
    total_size = embeddings_size + positional_enc_size + docs_size + 
                doc_embeddings_size + vector_store_size
    
    return Dict{String, Any}(
        "total_memory_mb" => total_size / (1024^2),
        "word_embeddings_mb" => embeddings_size / (1024^2),
        "positional_encoding_mb" => positional_enc_size / (1024^2),
        "documents_mb" => docs_size / (1024^2),
        "doc_embeddings_mb" => doc_embeddings_size / (1024^2),
        "vector_store_mb" => vector_store_size / (1024^2),
        "document_count" => length(ai.docs.documents),
        "vocabulary_size" => length(ai.vocab.word_to_idx),
        "vector_chunks" => length(vector_store.chunks),
        "gc_time_seconds" => Base.gc_num().total_time / 1e9,
        "allocated_memory_mb" => Base.gc_num().allocd / (1024^2)
    )
end

# Comprehensive initialization function
function initialize_enhanced_abida(db_path::String="enhanced_abida.duckdb"; 
                                 config::TransformerConfig=DEFAULT_CONFIG,
                                 load_path::String="",
                                 documents_dir::String="",
                                 enable_logging::Bool=true,
                                 log_level::LogLevel=Logging.Info,
                                 tokenizer_path::String="",
                                 merges_path::String="")
    """
    Initialize enhanced Abida system with all features and comprehensive logging
    """
    if enable_logging
        logger = ConsoleLogger(stdout, log_level)
        global_logger(logger)
    end
    
    @info "Initializing Enhanced Abida System" db_path=db_path config=config
    
    # Initialize core AI
    ai = if !isempty(load_path) && isfile(load_path)
        @info "Loading from checkpoint" load_path=load_path
        load(load_path, config, db_path)
    else
        @info "Creating new AI instance"
        AGI(db_path, config)
    end
    
    # Initialize vector store
    @info "Initializing vector store" dimension=config.d_model
    vector_store = VectorStore(config.d_model)
    
    # Initialize tokenizer if paths provided
    tokenizer = if !isempty(tokenizer_path) && !isempty(merges_path) && 
                  isfile(tokenizer_path) && isfile(merges_path)
        @info "Loading BPE tokenizer" vocab_file=tokenizer_path merges_file=merges_path
        BPETokenizer(tokenizer_path, merges_path)
    else
        @info "No tokenizer specified, using default word-based tokenization"
        nothing
    end
    
    # Load documents if directory provided
    if !isempty(documents_dir) && isdir(documents_dir)
        @info "Loading documents from directory" directory=documents_dir
        strategy = SemanticChunking(0.7f0, 300, 75)  # Use semantic chunking for better quality
        load_documents_from_directory!(ai, vector_store, documents_dir, [".txt", ".md", ".pdf"], strategy)
    end
    
    # Get initial system statistics
    stats = get_memory_usage_stats(ai, vector_store)
    
    @info "Enhanced Abida initialization completed successfully" 
          vocab_size=length(ai.vocab.word_to_idx)
          documents=length(ai.docs.documents)
          vector_chunks=length(vector_store.chunks)
          memory_usage_mb=round(stats["total_memory_mb"], digits=2)
          tokenizer_loaded=tokenizer !== nothing
    
    return ai, vector_store, tokenizer
end

function save_enhanced_checkpoint(ai::AGI, vector_store::VectorStore, path::String, 
                                tokenizer=nothing, metadata::Dict{String, Any}=Dict{String, Any}())
    """
    Save enhanced checkpoint with vector store and tokenizer
    """
    try
        @info "Saving enhanced checkpoint" path=path
        
        checkpoint_data = Dict{String, Any}(
            # Core AI data
            "vocab_idx_to_word" => ai.vocab.idx_to_word,
            "vocab_word_to_idx" => ai.vocab.word_to_idx,
            "word_embeddings" => ai.word_embeddings.matrix,
            "positional_enc" => ai.positional_enc.matrix,
            "documents" => ai.docs.documents,
            "doc_embeddings" => ai.docs.embeddings,
            "config" => ai.config,
            
            # Vector store data
            "vector_store_chunks" => [(
                chunk.id, chunk.content, chunk.source, chunk.metadata,
                chunk.embedding, chunk.timestamp, chunk.chunk_index,
                chunk.total_chunks, chunk.parent_doc_id
            ) for chunk in vector_store.chunks],
            "vector_store_config" => Dict{String, Any}(
                "dimension" => vector_store.dimension,
                "distance_metric" => string(vector_store.distance_metric),
                "index_type" => vector_store.index_type
            ),
            
            # Metadata
            "save_timestamp" => now(),
            "version" => "enhanced_1.0.0",
            "has_tokenizer" => tokenizer !== nothing,
            "user_metadata" => metadata
        )
        
        JLD2.jldopen(path, "w") do file
            for (key, value) in checkpoint_data
                file[key] = value
            end
        end
        
        @info "Enhanced checkpoint saved successfully" 
              path=path 
              size_mb=round(filesize(path) / (1024^2), digits=2)
              
    catch e
        @error "Failed to save enhanced checkpoint" path=path exception=e
        rethrow(e)
    end
end

function load_enhanced_checkpoint(path::String, config::TransformerConfig, db_path::String)
    """
    Load enhanced checkpoint with vector store
    """
    try
        @info "Loading enhanced checkpoint" path=path
        
        # Load checkpoint data
        checkpoint_data = JLD2.jldopen(path, "r") do file
            Dict{String, Any}(
                "vocab_idx_to_word" => file["vocab_idx_to_word"],
                "vocab_word_to_idx" => file["vocab_word_to_idx"],
                "word_embeddings" => file["word_embeddings"],
                "positional_enc" => file["positional_enc"],
                "documents" => file["documents"],
                "doc_embeddings" => file["doc_embeddings"],
                "vector_store_chunks" => get(file, "vector_store_chunks", []),
                "vector_store_config" => get(file, "vector_store_config", Dict{String, Any}()),
                "save_timestamp" => get(file, "save_timestamp", now()),
                "version" => get(file, "version", "unknown"),
                "has_tokenizer" => get(file, "has_tokenizer", false)
            )
        end
        
        # Reconstruct AI
        vocab = Vocabulary(checkpoint_data["vocab_word_to_idx"], checkpoint_data["vocab_idx_to_word"])
        
        db = DuckDB.DB(db_path)
        conn = DuckDB.connect(db)
        init_database(db)
        
        ai = AGI(
            vocab,
            WordEmbeddings(checkpoint_data["word_embeddings"]),
            PositionalEncoding(checkpoint_data["positional_enc"]),
            DocumentStore(checkpoint_data["documents"], checkpoint_data["doc_embeddings"]),
            config,
            conn
        )
        
        # Reconstruct vector store
        vs_config = checkpoint_data["vector_store_config"]
        dimension = get(vs_config, "dimension", config.d_model)
        vector_store = VectorStore(dimension)
        
        # Restore chunks
        for chunk_data in checkpoint_data["vector_store_chunks"]
            chunk = DocumentChunk(
                chunk_data[1],  # id
                chunk_data[2],  # content
                chunk_data[3],  # source
                chunk_data[4],  # metadata
                chunk_data[5],  # embedding
                chunk_data[7],  # chunk_index
                chunk_data[8],  # total_chunks
                chunk_data[9]   # parent_doc_id
            )
            push!(vector_store.chunks, chunk)
        end
        
        # Rebuild vector store index
        rebuild_index!(vector_store)
        
        @info "Enhanced checkpoint loaded successfully" 
              version=checkpoint_data["version"]
              saved_at=checkpoint_data["save_timestamp"]
              documents=length(ai.docs.documents)
              chunks=length(vector_store.chunks)
        
        return ai, vector_store
        
    catch e
        @error "Failed to load enhanced checkpoint" path=path exception=e
        rethrow(e)
    end
end

# Export all enhanced core components
export GenerationConfig, generate_text, generate_with_prompt_template, generate_multiple_candidates
export sample_token, sample_with_constraints
export generate_answer, ConversationState, conversational_answer
export add_exchange!, build_conversation_context, update_user_preferences!, get_conversation_summary
export learn_from_feedback!, analyze_feedback_patterns
export EvaluationResult, evaluate_model, evaluate_semantic_similarity, generate_evaluation_report
export optimize_memory!, get_memory_usage_stats
export initialize_enhanced_abida, save_enhanced_checkpoint, load_enhanced_checkpoint