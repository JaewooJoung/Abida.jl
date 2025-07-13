# web_api.jl - HTTP API server for Abida with OpenAI compatibility

using HTTP
using JSON3
using Dates
using Base64
using UUIDs
using Sockets
using WebSockets
using Logging
using OrderedCollections

# API Configuration
mutable struct APIConfig
    port::Int
    host::String
    max_tokens::Int
    timeout_seconds::Int
    enable_cors::Bool
    api_key::String
    rate_limit_requests_per_minute::Int
    max_request_size_mb::Int
    enable_streaming::Bool
    enable_websockets::Bool
    websocket_port::Int
    log_level::LogLevel
    enable_metrics::Bool
    admin_endpoints::Bool
end

function APIConfig(; 
    port=8080, 
    host="0.0.0.0", 
    max_tokens=1000, 
    timeout_seconds=30, 
    enable_cors=true, 
    api_key="",
    rate_limit_requests_per_minute=60,
    max_request_size_mb=10,
    enable_streaming=true,
    enable_websockets=true,
    websocket_port=8081,
    log_level=Logging.Info,
    enable_metrics=true,
    admin_endpoints=true
)
    APIConfig(port, host, max_tokens, timeout_seconds, enable_cors, api_key,
             rate_limit_requests_per_minute, max_request_size_mb, enable_streaming,
             enable_websockets, websocket_port, log_level, enable_metrics, admin_endpoints)
end

# Request rate limiting
mutable struct RateLimiter
    requests::Dict{String, Vector{DateTime}}
    max_requests::Int
    time_window_minutes::Int
end

function RateLimiter(max_requests::Int=60, time_window_minutes::Int=1)
    RateLimiter(Dict{String, Vector{DateTime}}(), max_requests, time_window_minutes)
end

function is_rate_limited(limiter::RateLimiter, client_ip::String)
    now_time = now()
    cutoff_time = now_time - Minute(limiter.time_window_minutes)
    
    # Clean old requests
    if haskey(limiter.requests, client_ip)
        filter!(t -> t > cutoff_time, limiter.requests[client_ip])
    else
        limiter.requests[client_ip] = DateTime[]
    end
    
    # Check if rate limited
    if length(limiter.requests[client_ip]) >= limiter.max_requests
        return true
    end
    
    # Add current request
    push!(limiter.requests[client_ip], now_time)
    return false
end

# Metrics collection
mutable struct APIMetrics
    total_requests::Int
    successful_requests::Int
    failed_requests::Int
    avg_response_time_ms::Float64
    response_times::Vector{Float64}
    endpoint_stats::Dict{String, Dict{String, Any}}
    start_time::DateTime
end

function APIMetrics()
    APIMetrics(0, 0, 0, 0.0, Float64[], Dict{String, Dict{String, Any}}(), now())
end

function record_request!(metrics::APIMetrics, endpoint::String, method::String, 
                        response_time_ms::Float64, success::Bool)
    metrics.total_requests += 1
    
    if success
        metrics.successful_requests += 1
    else
        metrics.failed_requests += 1
    end
    
    # Update response times (keep last 1000)
    push!(metrics.response_times, response_time_ms)
    if length(metrics.response_times) > 1000
        popfirst!(metrics.response_times)
    end
    
    metrics.avg_response_time_ms = mean(metrics.response_times)
    
    # Update endpoint-specific stats
    endpoint_key = "$method $endpoint"
    if !haskey(metrics.endpoint_stats, endpoint_key)
        metrics.endpoint_stats[endpoint_key] = Dict{String, Any}(
            "count" => 0,
            "avg_time" => 0.0,
            "success_rate" => 0.0,
            "times" => Float64[]
        )
    end
    
    stats = metrics.endpoint_stats[endpoint_key]
    stats["count"] += 1
    push!(stats["times"], response_time_ms)
    
    if length(stats["times"]) > 100
        popfirst!(stats["times"])
    end
    
    stats["avg_time"] = mean(stats["times"])
    stats["success_rate"] = success ? 
        (stats["success_rate"] * (stats["count"] - 1) + 1.0) / stats["count"] :
        (stats["success_rate"] * (stats["count"] - 1)) / stats["count"]
end

# Global state for the API server
mutable struct APIServer
    ai::AGI
    vector_store::VectorStore
    config::APIConfig
    server::Union{Nothing, HTTP.Server}
    websocket_server::Union{Nothing, HTTP.Server}
    start_time::DateTime
    metrics::APIMetrics
    rate_limiter::RateLimiter
    active_connections::Dict{String, Any}
    tokenizer::Union{Nothing, BPETokenizer}
end

function APIServer(ai::AGI, vector_store::VectorStore, config::APIConfig, tokenizer=nothing)
    APIServer(ai, vector_store, config, nothing, nothing, now(), 
             APIMetrics(), RateLimiter(config.rate_limit_requests_per_minute),
             Dict{String, Any}(), tokenizer)
end

# Utility functions
function get_client_ip(req::HTTP.Request)
    # Check various headers for real IP
    for header in ["X-Forwarded-For", "X-Real-IP", "CF-Connecting-IP"]
        ip = HTTP.header(req, header, "")
        if !isempty(ip)
            return split(ip, ",")[1] |> strip
        end
    end
    return "unknown"
end

function validate_request_size(req::HTTP.Request, max_size_mb::Int)
    content_length = parse(Int, HTTP.header(req, "Content-Length", "0"))
    max_bytes = max_size_mb * 1024 * 1024
    return content_length <= max_bytes
end

function parse_request_body(req::HTTP.Request)
    try
        return JSON3.read(IOBuffer(req.body))
    catch e
        @warn "Failed to parse request body" exception=e
        return nothing
    end
end

function create_error_response(status::Int, error::String, details::String="")
    error_data = Dict{String, Any}(
        "error" => Dict{String, Any}(
            "message" => error,
            "type" => "invalid_request_error",
            "code" => string(status)
        )
    )
    
    if !isempty(details)
        error_data["error"]["details"] = details
    end
    
    return HTTP.Response(status, [("Content-Type", "application/json")], 
                        JSON3.write(error_data))
end

function create_success_response(data::Dict, status::Int=200)
    return HTTP.Response(status, [("Content-Type", "application/json")], 
                        JSON3.write(data))
end

# Middleware functions
function cors_middleware(handler, config::APIConfig)
    return function(req::HTTP.Request)
        if !config.enable_cors
            return handler(req)
        end
        
        headers = [
            "Access-Control-Allow-Origin" => "*",
            "Access-Control-Allow-Methods" => "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers" => "Content-Type, Authorization, X-API-Key",
            "Access-Control-Max-Age" => "86400"
        ]
        
        # Handle preflight requests
        if req.method == "OPTIONS"
            return HTTP.Response(200, headers, "")
        end
        
        # Process the request
        response = handler(req)
        
        # Add CORS headers to response
        for (key, value) in headers
            HTTP.setheader(response, key => value)
        end
        
        return response
    end
end

function auth_middleware(handler, config::APIConfig)
    return function(req::HTTP.Request)
        if isempty(config.api_key)
            return handler(req)
        end
        
        # Check Authorization header
        auth_header = HTTP.header(req, "Authorization", "")
        api_key_header = HTTP.header(req, "X-API-Key", "")
        
        valid_auth = false
        if !isempty(auth_header) && startswith(auth_header, "Bearer ")
            token = auth_header[8:end]
            valid_auth = token == config.api_key
        elseif !isempty(api_key_header)
            valid_auth = api_key_header == config.api_key
        end
        
        if !valid_auth
            return create_error_response(401, "Invalid API key")
        end
        
        return handler(req)
    end
end

function rate_limit_middleware(handler, server::APIServer)
    return function(req::HTTP.Request)
        client_ip = get_client_ip(req)
        
        if is_rate_limited(server.rate_limiter, client_ip)
            return create_error_response(429, "Rate limit exceeded", 
                                       "Too many requests from this IP address")
        end
        
        return handler(req)
    end
end

function request_validation_middleware(handler, config::APIConfig)
    return function(req::HTTP.Request)
        # Validate request size
        if !validate_request_size(req, config.max_request_size_mb)
            return create_error_response(413, "Request too large")
        end
        
        # Validate Content-Type for POST/PUT requests
        if req.method in ["POST", "PUT"]
            content_type = HTTP.header(req, "Content-Type", "")
            if !occursin("application/json", content_type)
                return create_error_response(415, "Unsupported media type", 
                                           "Content-Type must be application/json")
            end
        end
        
        return handler(req)
    end
end

function logging_middleware(handler, server::APIServer)
    return function(req::HTTP.Request)
        start_time = time()
        client_ip = get_client_ip(req)
        request_id = string(uuid4())[1:8]
        
        @info "API Request" id=request_id method=req.method url=req.target client_ip=client_ip
        
        response = handler(req)
        
        duration_ms = (time() - start_time) * 1000
        success = 200 <= response.status < 400
        
        @info "API Response" id=request_id status=response.status duration_ms=round(duration_ms, digits=2)
        
        # Record metrics
        if server.config.enable_metrics
            record_request!(server.metrics, req.target, req.method, duration_ms, success)
        end
        
        return response
    end
end

# OpenAI-compatible endpoints
function handle_chat_completions(server::APIServer, req::HTTP.Request)
    try
        body = parse_request_body(req)
        if body === nothing
            return create_error_response(400, "Invalid JSON in request body")
        end
        
        messages = get(body, :messages, [])
        model = get(body, :model, "abida")
        max_tokens = get(body, :max_tokens, server.config.max_tokens)
        temperature = get(body, :temperature, 0.7)
        stream = get(body, :stream, false)
        
        if isempty(messages)
            return create_error_response(400, "No messages provided")
        end
        
        # Extract the conversation
        conversation_history = String[]
        for message in messages
            role = get(message, :role, "")
            content = get(message, :content, "")
            
            if !isempty(content)
                push!(conversation_history, "$role: $content")
            end
        end
        
        # Get the last user message
        last_message = messages[end]
        question = get(last_message, :content, "")
        
        if isempty(question)
            return create_error_response(400, "Empty message content")
        end
        
        # Generate response using comprehensive answer
        if stream && server.config.enable_streaming
            return handle_streaming_response(server, question, model, max_tokens, temperature)
        else
            return handle_non_streaming_response(server, question, conversation_history, 
                                               model, max_tokens, temperature)
        end
        
    catch e
        @error "Error in chat completions" exception=e
        return create_error_response(500, "Internal server error", string(e))
    end
end

function handle_non_streaming_response(server::APIServer, question::String, 
                                     conversation_history::Vector{String},
                                     model::String, max_tokens::Int, temperature::Float64)
    # Use comprehensive answer system
    answer_text, confidence, source = comprehensive_answer(server.ai, server.vector_store, question)
    
    # Trim response if too long
    if length(answer_text) > max_tokens * 4  # Rough token estimation
        answer_text = answer_text[1:max_tokens*4] * "..."
    end
    
    # Create OpenAI-compatible response
    response_data = Dict{String, Any}(
        "id" => "chatcmpl-" * string(uuid4()),
        "object" => "chat.completion",
        "created" => Int(time()),
        "model" => model,
        "choices" => [
            Dict{String, Any}(
                "index" => 0,
                "message" => Dict{String, Any}(
                    "role" => "assistant",
                    "content" => answer_text
                ),
                "finish_reason" => "stop"
            )
        ],
        "usage" => Dict{String, Any}(
            "prompt_tokens" => estimate_tokens(question),
            "completion_tokens" => estimate_tokens(answer_text),
            "total_tokens" => estimate_tokens(question) + estimate_tokens(answer_text)
        ),
        "metadata" => Dict{String, Any}(
            "confidence" => confidence,
            "source" => source,
            "conversation_length" => length(conversation_history)
        )
    )
    
    return create_success_response(response_data)
end

function handle_streaming_response(server::APIServer, question::String, 
                                 model::String, max_tokens::Int, temperature::Float64)
    # For streaming, we'll simulate word-by-word generation
    answer_text, confidence, source = comprehensive_answer(server.ai, server.vector_store, question)
    
    # Create streaming response
    response_headers = [
        ("Content-Type", "text/event-stream"),
        ("Cache-Control", "no-cache"),
        ("Connection", "keep-alive")
    ]
    
    return HTTP.Response(200, response_headers, 
        generate_streaming_body(answer_text, model, confidence, source))
end

function generate_streaming_body(answer_text::String, model::String, 
                                confidence::Float32, source::String)
    words = split(answer_text)
    chunks = String[]
    
    # Send initial chunk
    initial_chunk = Dict{String, Any}(
        "id" => "chatcmpl-" * string(uuid4()),
        "object" => "chat.completion.chunk",
        "created" => Int(time()),
        "model" => model,
        "choices" => [
            Dict{String, Any}(
                "index" => 0,
                "delta" => Dict{String, Any}("role" => "assistant"),
                "finish_reason" => nothing
            )
        ]
    )
    push!(chunks, "data: " * JSON3.write(initial_chunk) * "\n\n")
    
    # Send word chunks
    for (i, word) in enumerate(words)
        chunk = Dict{String, Any}(
            "id" => "chatcmpl-" * string(uuid4()),
            "object" => "chat.completion.chunk",
            "created" => Int(time()),
            "model" => model,
            "choices" => [
                Dict{String, Any}(
                    "index" => 0,
                    "delta" => Dict{String, Any}("content" => word * " "),
                    "finish_reason" => nothing
                )
            ]
        )
        push!(chunks, "data: " * JSON3.write(chunk) * "\n\n")
    end
    
    # Send final chunk
    final_chunk = Dict{String, Any}(
        "id" => "chatcmpl-" * string(uuid4()),
        "object" => "chat.completion.chunk",
        "created" => Int(time()),
        "model" => model,
        "choices" => [
            Dict{String, Any}(
                "index" => 0,
                "delta" => Dict{String, Any}(),
                "finish_reason" => "stop"
            )
        ]
    )
    push!(chunks, "data: " * JSON3.write(final_chunk) * "\n\n")
    push!(chunks, "data: [DONE]\n\n")
    
    return join(chunks, "")
end

function estimate_tokens(text::String)
    # Rough token estimation (about 4 characters per token on average)
    return max(1, length(text) ÷ 4)
end

# Simple completion endpoint (non-chat)
function handle_completions(server::APIServer, req::HTTP.Request)
    try
        body = parse_request_body(req)
        if body === nothing
            return create_error_response(400, "Invalid JSON in request body")
        end
        
        prompt = get(body, :prompt, "")
        max_tokens = get(body, :max_tokens, server.config.max_tokens)
        temperature = get(body, :temperature, 0.7)
        model = get(body, :model, "abida")
        
        if isempty(prompt)
            return create_error_response(400, "No prompt provided")
        end
        
        # Generate completion
        if server.tokenizer !== nothing
            generated = generate_text(server.ai, prompt, max_tokens=max_tokens, 
                                    temperature=Float32(temperature))
        else
            # Use RAG system for completion
            answer_text, confidence, source = comprehensive_answer(server.ai, server.vector_store, prompt)
            generated = answer_text
        end
        
        response_data = Dict{String, Any}(
            "id" => "cmpl-" * string(uuid4()),
            "object" => "text_completion",
            "created" => Int(time()),
            "model" => model,
            "choices" => [
                Dict{String, Any}(
                    "text" => generated,
                    "index" => 0,
                    "logprobs" => nothing,
                    "finish_reason" => "stop"
                )
            ],
            "usage" => Dict{String, Any}(
                "prompt_tokens" => estimate_tokens(prompt),
                "completion_tokens" => estimate_tokens(generated),
                "total_tokens" => estimate_tokens(prompt) + estimate_tokens(generated)
            )
        )
        
        return create_success_response(response_data)
        
    catch e
        @error "Error in completions" exception=e
        return create_error_response(500, "Internal server error", string(e))
    end
end

# Embeddings endpoint
function handle_embeddings(server::APIServer, req::HTTP.Request)
    try
        body = parse_request_body(req)
        if body === nothing
            return create_error_response(400, "Invalid JSON in request body")
        end
        
        input_text = get(body, :input, "")
        model = get(body, :model, "abida-embeddings")
        
        if isempty(input_text)
            return create_error_response(400, "No input text provided")
        end
        
        # Handle both string and array inputs
        texts = isa(input_text, String) ? [input_text] : input_text
        
        embeddings_data = []
        total_tokens = 0
        
        for (i, text) in enumerate(texts)
            # Generate embedding
            text_embeddings = encode_text(server.ai, text)
            embedding_vector = normalize(vec(mean(text_embeddings, dims=2)))
            
            push!(embeddings_data, Dict{String, Any}(
                "object" => "embedding",
                "index" => i - 1,
                "embedding" => Vector{Float64}(embedding_vector)
            ))
            
            total_tokens += estimate_tokens(text)
        end
        
        response_data = Dict{String, Any}(
            "object" => "list",
            "data" => embeddings_data,
            "model" => model,
            "usage" => Dict{String, Any}(
                "prompt_tokens" => total_tokens,
                "total_tokens" => total_tokens
            )
        )
        
        return create_success_response(response_data)
        
    catch e
        @error "Error in embeddings" exception=e
        return create_error_response(500, "Internal server error", string(e))
    end
end

# Knowledge management endpoints
function handle_learn(server::APIServer, req::HTTP.Request)
    try
        body = parse_request_body(req)
        if body === nothing
            return create_error_response(400, "Invalid JSON in request body")
        end
        
        text = get(body, :text, "")
        source = get(body, :source, "api")
        metadata = get(body, :metadata, Dict{String, Any}())
        chunking_strategy = get(body, :chunking_strategy, "fixed")
        
        if isempty(text)
            return create_error_response(400, "No text provided")
        end
        
        # Parse chunking strategy
        strategy = if chunking_strategy == "semantic"
            SemanticChunking(0.7f0, 300, 50)
        elseif chunking_strategy == "sentence"
            SentenceChunking(5, 1)
        elseif chunking_strategy == "paragraph"
            ParagraphChunking(3, 1)
        else
            FixedSizeChunking(200, 50)
        end
        
        # Add to knowledge base
        chunks_before = length(server.vector_store.chunks)
        update_knowledge!(server.ai, server.vector_store, text, source, strategy)
        chunks_added = length(server.vector_store.chunks) - chunks_before
        
        response_data = Dict{String, Any}(
            "success" => true,
            "message" => "Knowledge updated successfully",
            "chunks_added" => chunks_added,
            "total_chunks" => length(server.vector_store.chunks),
            "source" => source,
            "strategy" => chunking_strategy
        )
        
        return create_success_response(response_data)
        
    catch e
        @error "Error in learn endpoint" exception=e
        return create_error_response(500, "Internal server error", string(e))
    end
end

function handle_search(server::APIServer, req::HTTP.Request)
    try
        # Parse query parameters
        uri = HTTP.URI(req.target)
        query_params = HTTP.queryparams(uri)
        
        query = get(query_params, "q", "")
        k = parse(Int, get(query_params, "k", "5"))
        search_type = get(query_params, "type", "hybrid")  # semantic, keyword, hybrid
        threshold = parse(Float32, get(query_params, "threshold", "0.1"))
        
        if isempty(query)
            return create_error_response(400, "No query provided")
        end
        
        # Perform search based on type
        results = if search_type == "semantic"
            semantic_search(server.ai, server.vector_store, query, k, threshold)
        elseif search_type == "keyword"
            keyword_search(server.vector_store, query, k)
        else  # hybrid
            hybrid_search(server.ai, server.vector_store, query, k, 0.7f0)
        end
        
        # Format results
        formatted_results = [
            Dict{String, Any}(
                "id" => chunk.id,
                "content" => chunk.content,
                "source" => chunk.source,
                "metadata" => chunk.metadata,
                "timestamp" => chunk.timestamp,
                "chunk_index" => chunk.chunk_index,
                "total_chunks" => chunk.total_chunks
            ) for chunk in results
        ]
        
        response_data = Dict{String, Any}(
            "query" => query,
            "search_type" => search_type,
            "results" => formatted_results,
            "total_results" => length(results),
            "parameters" => Dict{String, Any}(
                "k" => k,
                "threshold" => threshold
            )
        )
        
        return create_success_response(response_data)
        
    catch e
        @error "Error in search endpoint" exception=e
        return create_error_response(500, "Internal server error", string(e))
    end
end

function handle_upload_document(server::APIServer, req::HTTP.Request)
    try
        # This would handle multipart/form-data uploads
        # For now, expecting JSON with base64 encoded content
        body = parse_request_body(req)
        if body === nothing
            return create_error_response(400, "Invalid JSON in request body")
        end
        
        filename = get(body, :filename, "uploaded_file")
        content_base64 = get(body, :content, "")
        source = get(body, :source, filename)
        metadata = get(body, :metadata, Dict{String, Any}())
        
        if isempty(content_base64)
            return create_error_response(400, "No content provided")
        end
        
        # Decode base64 content
        try
            content = String(base64decode(content_base64))
        catch
            return create_error_response(400, "Invalid base64 content")
        end
        
        # Add metadata about the upload
        metadata["uploaded_at"] = now()
        metadata["filename"] = filename
        metadata["upload_method"] = "api"
        
        # Add to knowledge base
        chunks_before = length(server.vector_store.chunks)
        add_document!(server.ai, server.vector_store, content, source, metadata)
        chunks_added = length(server.vector_store.chunks) - chunks_before
        
        response_data = Dict{String, Any}(
            "success" => true,
            "message" => "Document uploaded successfully",
            "filename" => filename,
            "chunks_added" => chunks_added,
            "total_chunks" => length(server.vector_store.chunks)
        )
        
        return create_success_response(response_data, 201)
        
    catch e
        @error "Error in upload endpoint" exception=e
        return create_error_response(500, "Internal server error", string(e))
    end
end

# Admin and monitoring endpoints
function handle_health(server::APIServer, req::HTTP.Request)
    uptime_seconds = Dates.value(now() - server.start_time) ÷ 1000
    
    # Check system health
    memory_usage = Base.gc_num().total_time / 1e9  # Convert to seconds
    
    health_status = "healthy"
    if memory_usage > 300  # More than 5 minutes of GC time indicates issues
        health_status = "degraded"
    end
    
    status_data = Dict{String, Any}(
        "status" => health_status,
        "uptime_seconds" => uptime_seconds,
        "uptime_human" => format_duration(uptime_seconds),
        "memory" => Dict{String, Any}(
            "gc_time_seconds" => memory_usage,
            "allocated_mb" => Base.gc_num().allocd / (1024^2)
        ),
        "knowledge_stats" => Dict{String, Any}(
            "documents" => length(server.ai.docs.documents),
            "vocabulary_size" => length(server.ai.vocab.word_to_idx),
            "vector_store_chunks" => length(server.vector_store.chunks),
            "unique_sources" => length(unique([chunk.source for chunk in server.vector_store.chunks]))
        ),
        "api_stats" => if server.config.enable_metrics
            Dict{String, Any}(
                "total_requests" => server.metrics.total_requests,
                "success_rate" => server.metrics.total_requests > 0 ? 
                    server.metrics.successful_requests / server.metrics.total_requests : 0.0,
                "avg_response_time_ms" => server.metrics.avg_response_time_ms
            )
        else
            Dict{String, Any}("metrics_disabled" => true)
        end,
        "version" => "1.0.0",
        "features" => Dict{String, Any}(
            "streaming" => server.config.enable_streaming,
            "websockets" => server.config.enable_websockets,
            "rate_limiting" => server.config.rate_limit_requests_per_minute > 0,
            "authentication" => !isempty(server.config.api_key)
        )
    )
    
    status_code = health_status == "healthy" ? 200 : 503
    return HTTP.Response(status_code, [("Content-Type", "application/json")], 
                        JSON3.write(status_data))
end

function handle_metrics(server::APIServer, req::HTTP.Request)
    if !server.config.enable_metrics
        return create_error_response(404, "Metrics not enabled")
    end
    
    metrics_data = Dict{String, Any}(
        "requests" => Dict{String, Any}(
            "total" => server.metrics.total_requests,
            "successful" => server.metrics.successful_requests,
            "failed" => server.metrics.failed_requests,
            "success_rate" => server.metrics.total_requests > 0 ? 
                server.metrics.successful_requests / server.metrics.total_requests : 0.0
        ),
        "performance" => Dict{String, Any}(
            "avg_response_time_ms" => server.metrics.avg_response_time_ms,
            "response_time_p50" => length(server.metrics.response_times) > 0 ? 
                quantile(server.metrics.response_times, 0.5) : 0.0,
            "response_time_p95" => length(server.metrics.response_times) > 0 ? 
                quantile(server.metrics.response_times, 0.95) : 0.0,
            "response_time_p99" => length(server.metrics.response_times) > 0 ? 
                quantile(server.metrics.response_times, 0.99) : 0.0
        ),
        "endpoints" => server.metrics.endpoint_stats,
        "rate_limiting" => Dict{String, Any}(
            "active_clients" => length(server.rate_limiter.requests),
            "requests_per_minute_limit" => server.rate_limiter.max_requests
        ),
        "collection_period" => Dict{String, Any}(
            "start_time" => server.metrics.start_time,
            "duration_seconds" => Dates.value(now() - server.metrics.start_time) ÷ 1000
        )
    )
    
    return create_success_response(metrics_data)
end

function handle_admin_optimize(server::APIServer, req::HTTP.Request)
    if !server.config.admin_endpoints
        return create_error_response(404, "Admin endpoints not enabled")
    end
    
    try
        # Optimize vector store
        chunks_before = length(server.vector_store.chunks)
        optimize_vector_store!(server.vector_store, 0.95f0)
        chunks_after = length(server.vector_store.chunks)
        
        # Force garbage collection
        GC.gc()
        
        response_data = Dict{String, Any}(
            "success" => true,
            "message" => "System optimization completed",
            "chunks_removed" => chunks_before - chunks_after,
            "chunks_remaining" => chunks_after,
            "memory_freed" => "GC performed"
        )
        
        return create_success_response(response_data)
        
    catch e
        @error "Error in admin optimize" exception=e
        return create_error_response(500, "Optimization failed", string(e))
    end
end

function handle_models(server::APIServer, req::HTTP.Request)
    models_data = Dict{String, Any}(
        "object" => "list",
        "data" => [
            Dict{String, Any}(
                "id" => "abida",
                "object" => "model",
                "created" => Int(time()),
                "owned_by" => "abida-ai",
                "permission" => [],
                "root" => "abida",
                "parent" => nothing
            ),
            Dict{String, Any}(
                "id" => "abida-embeddings",
                "object" => "model", 
                "created" => Int(time()),
                "owned_by" => "abida-ai",
                "permission" => [],
                "root" => "abida-embeddings",
                "parent" => nothing
            )
        ]
    )
    
    return create_success_response(models_data)
end

# WebSocket handling
function handle_websocket_connection(server::APIServer, ws)
    client_id = string(uuid4())[1:8]
    server.active_connections[client_id] = Dict{String, Any}(
        "websocket" => ws,
        "connected_at" => now(),
        "message_count" => 0
    )
    
    @info "WebSocket client connected" client_id=client_id
    
    try
        # Send welcome message
        welcome_msg = Dict{String, Any}(
            "type" => "welcome",
            "client_id" => client_id,
            "server_time" => now(),
            "features" => ["chat", "search", "learn"]
        )
        WebSockets.send(ws, JSON3.write(welcome_msg))
        
        # Handle incoming messages
        for msg in ws
            try
                data = JSON3.read(String(msg))
                response = handle_websocket_message(server, client_id, data)
                
                if response !== nothing
                    WebSockets.send(ws, JSON3.write(response))
                end
                
                server.active_connections[client_id]["message_count"] += 1
                
            catch e
                @warn "Error processing WebSocket message" client_id=client_id exception=e
                error_response = Dict{String, Any}(
                    "type" => "error",
                    "message" => "Failed to process message",
                    "error" => string(e)
                )
                WebSockets.send(ws, JSON3.write(error_response))
            end
        end
        
    catch e
        @warn "WebSocket connection error" client_id=client_id exception=e
    finally
        delete!(server.active_connections, client_id)
        @info "WebSocket client disconnected" client_id=client_id
    end
end

function handle_websocket_message(server::APIServer, client_id::String, data::Dict)
    message_type = get(data, "type", "")
    
    if message_type == "chat"
        message = get(data, "message", "")
        if !isempty(message)
            answer, confidence, source = comprehensive_answer(server.ai, server.vector_store, message)
            
            return Dict{String, Any}(
                "type" => "chat_response",
                "message" => answer,
                "confidence" => confidence,
                "source" => source,
                "timestamp" => now()
            )
        end
        
    elseif message_type == "search"
        query = get(data, "query", "")
        k = get(data, "k", 5)
        
        if !isempty(query)
            results = hybrid_search(server.ai, server.vector_store, query, k)
            
            return Dict{String, Any}(
                "type" => "search_results",
                "query" => query,
                "results" => [
                    Dict{String, Any}(
                        "content" => chunk.content[1:min(200, length(chunk.content))] * "...",
                        "source" => chunk.source,
                        "chunk_index" => chunk.chunk_index
                    ) for chunk in results
                ],
                "total" => length(results),
                "timestamp" => now()
            )
        end
        
    elseif message_type == "learn"
        text = get(data, "text", "")
        source = get(data, "source", "websocket:$client_id")
        
        if !isempty(text)
            update_knowledge!(server.ai, server.vector_store, text, source)
            
            return Dict{String, Any}(
                "type" => "learn_response",
                "success" => true,
                "message" => "Knowledge updated successfully",
                "timestamp" => now()
            )
        end
        
    elseif message_type == "ping"
        return Dict{String, Any}(
            "type" => "pong",
            "timestamp" => now()
        )
    end
    
    return Dict{String, Any}(
        "type" => "error",
        "message" => "Unknown message type or invalid data"
    )
end

# Utility functions
function format_duration(seconds::Int)
    if seconds < 60
        return "$(seconds)s"
    elseif seconds < 3600
        return "$(seconds ÷ 60)m $(seconds % 60)s"
    elseif seconds < 86400
        hours = seconds ÷ 3600
        minutes = (seconds % 3600) ÷ 60
        return "$(hours)h $(minutes)m"
    else
        days = seconds ÷ 86400
        hours = (seconds % 86400) ÷ 3600
        return "$(days)d $(hours)h"
    end
end

# Request router
function route_request(server::APIServer, req::HTTP.Request)
    path = HTTP.URI(req.target).path
    method = req.method
    
    try
        if method == "POST"
            if path == "/v1/chat/completions"
                return handle_chat_completions(server, req)
            elseif path == "/v1/completions"
                return handle_completions(server, req)
            elseif path == "/v1/embeddings"
                return handle_embeddings(server, req)
            elseif path == "/v1/learn"
                return handle_learn(server, req)
            elseif path == "/v1/upload"
                return handle_upload_document(server, req)
            elseif path == "/admin/optimize"
                return handle_admin_optimize(server, req)
            end
            
        elseif method == "GET"
            if path == "/v1/search"
                return handle_search(server, req)
            elseif path == "/health"
                return handle_health(server, req)
            elseif path == "/metrics"
                return handle_metrics(server, req)
            elseif path == "/v1/models"
                return handle_models(server, req)
            elseif path == "/"
                return HTTP.Response(200, [("Content-Type", "text/html")], 
                                   generate_api_documentation())
            end
        end
        
        # 404 for unknown routes
        return create_error_response(404, "Route not found", 
                                   "Available endpoints: /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/learn, /v1/search, /v1/upload, /health, /metrics")
        
    catch e
        @error "Error in route handler" path=path method=method exception=e
        return create_error_response(500, "Internal server error", string(e))
    end
end

function generate_api_documentation()
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Abida API Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }
            .method { font-weight: bold; color: #fff; padding: 5px 10px; border-radius: 3px; }
            .post { background-color: #49cc90; }
            .get { background-color: #61affe; }
            code { background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Abida API Documentation</h1>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/v1/chat/completions</code>
            <p>OpenAI-compatible chat completions endpoint. Supports streaming.</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/v1/completions</code>
            <p>Text completion endpoint.</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/v1/embeddings</code>
            <p>Generate embeddings for text input.</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/v1/learn</code>
            <p>Add new knowledge to the system.</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/v1/search</code>
            <p>Search the knowledge base. Parameters: q (query), k (results), type (semantic/keyword/hybrid)</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/health</code>
            <p>System health and status information.</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/metrics</code>
            <p>API usage metrics and performance statistics.</p>
        </div>
        
        <p>WebSocket endpoint available at port """ * string(8081) * """ for real-time interactions.</p>
    </body>
    </html>
    """
end

# Start the servers
function start_server!(server::APIServer)
    @info "Starting Abida API server" port=server.config.port host=server.config.host
    
    # Build middleware stack
    handler = req -> route_request(server, req)
    handler = logging_middleware(handler, server)
    handler = request_validation_middleware(handler, server.config)
    handler = rate_limit_middleware(handler, server)
    handler = auth_middleware(handler, server.config)
    handler = cors_middleware(handler, server.config)
    
    # Start HTTP server
    server.server = HTTP.serve!(handler, server.config.host, server.config.port)
    
    # Start WebSocket server if enabled
    if server.config.enable_websockets
        start_websocket_server!(server)
    end
    
    @info "Abida API server started successfully" port=server.config.port websockets=server.config.enable_websockets
    
    print_startup_info(server)
end

function start_websocket_server!(server::APIServer)
    try
        @info "Starting WebSocket server" port=server.config.websocket_port
        
        server.websocket_server = WebSockets.serve!(server.config.host, server.config.websocket_port) do ws
            handle_websocket_connection(server, ws)
        end
        
        @info "WebSocket server started" port=server.config.websocket_port
    catch e
        @error "Failed to start WebSocket server" exception=e
    end
end

function stop_server!(server::APIServer)
    if server.server !== nothing
        @info "Stopping HTTP server"
        close(server.server)
        server.server = nothing
    end
    
    if server.websocket_server !== nothing
        @info "Stopping WebSocket server"
        close(server.websocket_server)
        server.websocket_server = nothing
    end
    
    @info "Abida API server stopped"
end

function print_startup_info(server::APIServer)
    host = server.config.host
    port = server.config.port
    ws_port = server.config.websocket_port
    
    println("\n" * "="^80)
    println("🚀 Abida API Server Started Successfully!")
    println("="^80)
    println("📍 Base URL: http://$host:$port")
    println("📖 Documentation: http://$host:$port/")
    println("❤️  Health Check: http://$host:$port/health")
    println("📊 Metrics: http://$host:$port/metrics")
    
    if server.config.enable_websockets
        println("🔌 WebSocket: ws://$host:$ws_port")
    end
    
    println("\n📡 OpenAI-Compatible Endpoints:")
    println("   POST /v1/chat/completions    - Chat completions")
    println("   POST /v1/completions         - Text completions")
    println("   POST /v1/embeddings          - Generate embeddings")
    println("   GET  /v1/models              - List available models")
    
    println("\n🧠 Knowledge Management:")
    println("   POST /v1/learn               - Add knowledge")
    println("   GET  /v1/search              - Search knowledge")
    println("   POST /v1/upload              - Upload documents")
    
    if !isempty(server.config.api_key)
        println("\n🔐 Authentication: API key required (Bearer token or X-API-Key header)")
    end
    
    if server.config.rate_limit_requests_per_minute > 0
        println("⚡ Rate Limit: $(server.config.rate_limit_requests_per_minute) requests/minute per IP")
    end
    
    println("\n📊 Current Knowledge Base:")
    println("   Documents: $(length(server.ai.docs.documents))")
    println("   Vector Chunks: $(length(server.vector_store.chunks))")
    println("   Vocabulary: $(length(server.ai.vocab.word_to_idx)) tokens")
    
    println("\n🔧 Example Usage:")
    println("curl -X POST http://$host:$port/v1/chat/completions \\")
    println("  -H 'Content-Type: application/json' \\")
    if !isempty(server.config.api_key)
        println("  -H 'Authorization: Bearer YOUR_API_KEY' \\")
    end
    println("  -d '{")
    println("    \"model\": \"abida\",")
    println("    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],")
    println("    \"stream\": true")
    println("  }'")
    
    println("="^80 * "\n")
end

# Convenience function to start everything
function serve_abida(ai::AGI, vector_store::VectorStore; 
                    port::Int=8080, 
                    api_key::String="", 
                    enable_websockets::Bool=true,
                    tokenizer=nothing,
                    kwargs...)
    
    config = APIConfig(; port=port, api_key=api_key, enable_websockets=enable_websockets, kwargs...)
    server = APIServer(ai, vector_store, config, tokenizer)
    start_server!(server)
    return server
end

# Export API components
export APIConfig, APIServer, serve_abida, start_server!, stop_server!
export RateLimiter, APIMetrics
export handle_chat_completions, handle_completions, handle_embeddings
export handle_learn, handle_search, handle_upload_document
export handle_health, handle_metrics, handle_models