* `$` 로 둘러쌓인 언더바 : `(?<=\$[^$]*)_(?=[^$]*\$)`
  * `\\_` -> `\_`
* `$$`뒤에 개행안된 것 찾기 `\$\$(?=[^\n])`
* `$$`앞에 문자 있는 것 찾기 `(?<=[^\n])\$\$`
