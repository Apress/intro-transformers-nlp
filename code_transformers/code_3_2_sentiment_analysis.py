
model = pipeline('sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")
data = ["idk about you guys but i'm having more fun during the bear than I was having in the bull.","At least in the bear market it’s down only. Bull market is up and down"]
model(data)
