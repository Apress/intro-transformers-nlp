text = """ 
String google = "http://ajax.googleapis.com/ajax/services/search/web?v=1.0&q=";
    String search = "stackoverflow";
    String charset = "UTF-8";
    
    URL url = new URL(google + URLEncoder.encode(search, charset));
    Reader reader = new InputStreamReader(url.openStream(), charset);
    GoogleResults results = new Gson().fromJson(reader, GoogleResults.class);
    
    // Show title and URL of 1st result.
    System.out.println(results.getResponseData().getResults().get(0).getTitle());
    System.out.println(results.getResponseData().getResults().get(0).getUrl());
"""

input_ids = codeT5_tkn(text, return_tensors="pt").input_ids
gen_ids = mdl.generate(input_ids, max_length=50, temperature=0.2,num_beams=200,no_repeat_ngram_size=2,num_return_sequences=5)

print(codeT5_tkn.decode(gen_ids[0], skip_special_tokens=True))
