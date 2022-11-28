from transformers import RobertaTokenizer, T5ForConditionalGeneration

model_name_or_path = './comment_model'  #  Path to the folder created earlier.

codeT5_tkn = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
mdl = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

# provide code snippet as input
text = """ public static void main(String[] args) {

    int num = 29;
    boolean flag = false;
    for (int i = 2; i <= num / 2; ++i) {
      // condition for nonprime number
      if (num % i == 0) {
        flag = true;
        break;
      }
    }

    if (!flag)
      System.out.println(num + " is a prime number.");
    else
      System.out.println(num + " is not a prime number.");
  } """

input_ids = codeT5_tkn(text, return_tensors="pt").input_ids
gen_ids = mdl.generate(input_ids, max_length=20)

print(codeT5_tkn.decode(gen_ids[0], skip_special_tokens=True))
