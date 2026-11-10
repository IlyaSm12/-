!pip install gigachat -q
!curl -k "https://gu-st.ru/content/Other/doc/russian_trusted_root_ca.cer" -w "\n" >> $(python -m certifi)
