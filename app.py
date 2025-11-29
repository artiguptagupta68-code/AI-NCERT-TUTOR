# Generate answer (CPU only)
input_text = f"Answer the question based on the following context:\n{context}\nQuestion: {query}"

# Send tokens to CPU (not CUDA)
inputs = tokenizer(input_text, return_tensors="pt")

# Force model to use CPU
llm_model.to("cpu")

with torch.no_grad():
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

st.subheader("Answer:")
st.write(answer)
