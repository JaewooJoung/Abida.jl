🚀 To Make It FASTER:
julia# Maximum speed settings
const MAX_CONCURRENT_REQUESTS = 20  # More parallel connections
const BATCH_SIZE = 30               # Process more words at once
const WIKI_ARTICLES_PER_WORD = 2    # Fewer Wikipedia articles
const MIN_SENTENCE_LENGTH = 30      # Learn only longer sentences
const DELAY_BETWEEN_BATCHES = 0     # No delays
🐢 To Make It SLOWER/More Stable:
julia# Conservative settings
const MAX_CONCURRENT_REQUESTS = 5   # Fewer connections
const BATCH_SIZE = 5                # Smaller batches
const WIKI_ARTICLES_PER_WORD = 10   # More thorough learning
const MIN_SENTENCE_LENGTH = 10      # Learn more sentences
const DELAY_BETWEEN_BATCHES = 5     # Longer breaks
📊 To Change DURATION:
julia# Quick test (10 minutes)
const TOTAL_WORDS_TO_LEARN = 100
const WORDS_PER_MEGABATCH = 20

# Medium session (1 hour)
const TOTAL_WORDS_TO_LEARN = 500
const WORDS_PER_MEGABATCH = 50

# Full learning (2-3 hours)
const TOTAL_WORDS_TO_LEARN = 1000
const WORDS_PER_MEGABATCH = 100

# Marathon session (5+ hours)
const TOTAL_WORDS_TO_LEARN = 5000
const WORDS_PER_MEGABATCH = 200
