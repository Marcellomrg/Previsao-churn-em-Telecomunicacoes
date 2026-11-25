CREATE TABLE "final" AS
SELECT enem.*,
       CASE WHEN censo.CO_ENTIDADE IS NULL THEN 0 ELSE 1 END AS escola_encontrada_no_censo,
       censo.*
FROM resumo_enem_escolar AS enem
LEFT JOIN censo_escolar AS censo
    ON censo.CO_ENTIDADE = enem.CO_ESCOLA
   AND censo.NU_ANO_CENSO = enem.NU_ANO;

CREATE UNIQUE INDEX final_chave ON "final" (NU_ANO, CO_ESCOLA);

