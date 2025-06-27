# test_learn.jl
using Abida

# Initialize AGI instance
db_path = "test_knowledge.duckdb"
agi = AGI(db_path)

# Sample facts to learn
facts = [
    # General Knowledge
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The Earth orbits the Sun once every 365.25 days.",
    "Humans have 23 pairs of chromosomes.",
    
    # Science & Technology
    "Artificial Intelligence involves machines performing tasks that typically require human intelligence.",
    "Machine learning is a subset of AI that uses algorithms to learn from data.",
    "Deep learning uses neural networks with many layers to model complex patterns.",
    "Julia is a high-performance programming language for technical computing.",
    "DuckDB is an in-process SQL database optimized for analytical queries.",
    
    # Math
    "The square root of 16 is 4.",
    "The derivative of x^2 is 2x.",
    "In a right triangle, the Pythagorean theorem states that a² + b² = c².",
    "Pi (π) is approximately equal to 3.14159.",
    "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.",
    
    # History
    "World War II ended in 1945.",
    "The United States Declaration of Independence was signed on July 4, 1776.",
    "The Industrial Revolution began in Britain during the 18th century.",
    "Napoleon Bonaparte was a French military leader and emperor who ruled much of Europe in the early 19th century.",
    "The Berlin Wall fell in 1989.",
    
    # Geography
    "Mount Everest is the tallest mountain in the world.",
    "Australia is both a continent and a country.",
    "The Nile River is the longest river in the world.",
    "Japan is an island nation located in East Asia.",
    "Brazil is the largest country in South America by area.",
    
    # Animals
    "Kangaroos are native to Australia.",
    "Sharks are cartilaginous fish known for their sharp teeth.",
    "Bees play a crucial role in pollination.",
    "Octopuses are highly intelligent marine animals with eight arms.",
    "Penguins are flightless birds commonly found in the Southern Hemisphere.",
    
    # Space
    "The Moon orbits Earth and causes tides.",
    "Mars is known as the Red Planet due to its reddish appearance.",
    "There are eight planets in our solar system.",
    "Saturn has a prominent ring system made mostly of ice and rock particles.",
    "The International Space Station orbits Earth about every 90 minutes.",
    
    # Literature
    "William Shakespeare wrote 'Hamlet', 'Romeo and Juliet', and 'Macbeth'.",
    "'To be or not to be' is a famous soliloquy from Shakespeare's Hamlet.",
    "George Orwell wrote the dystopian novels '1984' and 'Animal Farm'.",
    "J.R.R. Tolkien wrote 'The Lord of the Rings' series.",
    "Jane Austen wrote 'Pride and Prejudice'.",
    
    # Technology
    "HTTP stands for HyperText Transfer Protocol.",
    "HTML is used to structure web pages.",
    "CSS is used to style HTML elements.",
    "JavaScript is a scripting language commonly used in web development.",
    "Python is a popular programming language known for its readability.",
    
    # Random Facts
    "Bananas are berries, but strawberries aren't.",
    "Octopuses have three hearts.",
    "Honey never spoils if stored properly.",
    "Lightning strikes Earth over 100 times per second.",
    "There are more stars in the universe than grains of sand on Earth.",
    
    # Repeat pattern variations to reach ~100 total
    "The sun is a star at the center of our solar system.",
    "Albert Einstein developed the theory of relativity.",
    "Computers use binary code consisting of ones and zeros.",
    "The internet connects billions of devices worldwide.",
    "Electricity is the flow of electric charge through a conductor.",
    
    "Natural Language Processing helps computers understand human language.",
    "Reinforcement Learning is a type of machine learning where agents learn by trial and error.",
    "Neural networks are computing systems inspired by the human brain.",
    "Data science involves extracting insights from large amounts of data.",
    "Algorithms are step-by-step procedures for calculations.",
    
    "The moon affects Earth’s tides.",
    "Cats sleep an average of 12–16 hours a day.",
    "Bats use echolocation to navigate and find prey.",
    "Human DNA is about 98% similar to chimpanzee DNA.",
    "Antarctica is the coldest place on Earth.",
    
    "The Great Wall of China is visible from space.",
    "The Mona Lisa was painted by Leonardo da Vinci.",
    "Vincent van Gogh cut off part of his ear.",
    "The Sistine Chapel ceiling was painted by Michelangelo.",
    "Impressionism is an art movement that originated in France in the 19th century.",
    
    "The Amazon rainforest produces 20% of the world's oxygen.",
    "Coral reefs are among the most diverse ecosystems on Earth.",
    "Rainforests cover only 6% of Earth's surface.",
    "Over half of all animal and plant species live in rainforests.",
    "Rainforests help regulate global climate.",
    
    "The human brain weighs about 1.4 kilograms.",
    "The heart pumps blood throughout the body via the circulatory system.",
    "DNA carries genetic instructions used in growth, development, functioning, and reproduction.",
    "The liver performs over 500 essential functions in the body.",
    "The human skeleton has 206 bones.",

    "The Eiffel Tower is located in Paris, France.",
    "The Statue of Liberty was a gift from France to the United States.",
    "The Great Pyramid of Giza is the oldest of the Seven Wonders of the Ancient World.",
    "The Taj Mahal is a mausoleum located in Agra, India.",
    "The Colosseum in Rome was used for gladiatorial contests and public spectacles.",

    "Solar power converts energy from the sun into electricity.",
    "Wind turbines convert kinetic energy into electrical energy.",
    "Hydropower uses flowing water to generate electricity.",
    "Fossil fuels include coal, oil, and natural gas.",
    "Renewable energy comes from sources that are naturally replenished.",

    "The Internet of Things connects everyday objects to the internet.",
    "Blockchain is a decentralized digital ledger technology behind Bitcoin.",
    "Cloud computing allows data storage and processing over the internet.",
    "Quantum computing uses quantum bits (qubits) to process information.",
    "Edge computing processes data near the source instead of in a central data center.",

    "The Turing Test measures a machine's ability to exhibit intelligent behavior.",
    "AI ethics deals with moral issues in artificial intelligence.",
    "Self-driving cars use sensors and machine learning to navigate roads.",
    "Robots are programmable machines capable of carrying out complex actions.",
    "Computer vision enables machines to interpret visual information from the world.",

    "The Large Hadron Collider is the world's largest particle accelerator.",
    "Atoms are the basic building blocks of matter.",
    "Chemical elements are substances made from one type of atom.",
    "The periodic table organizes all known chemical elements.",
    "Oxygen makes up about 21% of Earth's atmosphere.",

    "Time zones divide the Earth into regions with uniform standard time.",
    "Leap years occur every 4 years and add an extra day in February.",
    "A millennium consists of 1,000 years.",
    "A decade consists of 10 years.",
    "A century consists of 100 years."
]

# Learn each fact
for (i, fact) in enumerate(facts)
    learn!(agi, fact)
    if i % 10 == 0
        println("Learned $i facts...")
    end
end

println("✅ Finished learning $(length(facts)) facts.")

# Optional: Save state
save(agi, "agi_test_state.jld2")
println("💾 Model state saved to agi_test_state.jld2")

# Cleanup
cleanup!(agi)
