extensions = dict(
    required_params=[('training_frame', "NULL")],
    validate_frames="""
# training_frame is required if pre_trained frame is not specified
if (missing(pre_trained) && missing(training_frame)) stop(\"argument \'training_frame\' is missing, with no default\")
training_frame <- .validate.H2OFrame(training_frame)
pre_trained <- .validate.H2OFrame(pre_trained)
""",
    validate_required_params="",
    set_required_params="""
parms$training_frame <- training_frame
""",
)

doc = dict(
    preamble="""
Trains a word2vec model on a String column of an H2O data frame
""",
    examples="""
library(h2o)
h2o.init()

# Import the CraigslistJobTitles dataset
job.titles.path = "https://raw.githubusercontent.com/h2oai/sparkling-water/rel-1.6/examples/smalldata/craigslistJobTitles.csv"
job.titles <- h2o.importFile(job.titles.path, destination_frame = "jobtitles",
                             col.names = c("category", "jobtitle"),
                             col.types = c("String", "String"), header = TRUE)

# Build and train the Word2Vec model
words <- h2o.tokenize(job.titles, " ")
vec <- h2o.word2vec(training_frame = words)
h2o.findSynonyms(vec, "teacher", count = 20)
"""
)
