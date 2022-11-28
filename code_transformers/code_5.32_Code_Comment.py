from transformers import RobertaTokenizer, T5ForConditionalGeneration

model_name_or_path = './comment_model'  #  Path to the folder created earlier.

codeT5_tkn = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
mdl = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

# provide code snippet as input
text = """ LocalDate localDate = new LocalDate(2020, 1, 31);
int numberOfDays = Days.daysBetween(localDate, localDate.plusYears(1)).getDays();

boolean isLeapYear = (numberOfDays > 365) ? true : false;"""

input_ids = codeT5_tkn(text, return_tensors="pt").input_ids
gen_ids = mdl.generate(input_ids, max_length=150)

print(codeT5_tkn.decode(gen_ids[0], skip_special_tokens=True))
