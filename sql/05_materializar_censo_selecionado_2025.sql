-- O script de montagem insere os campos do ranking no marcador abaixo.
CREATE TABLE censo_escolar AS
SELECT Escola.NU_ANO_CENSO, Escola.CO_ENTIDADE,
       /* CAMPOS_SELECIONADOS */
FROM origem.Escola
LEFT JOIN origem.Matricula USING (NU_ANO_CENSO, CO_ENTIDADE)
LEFT JOIN origem.Docente USING (NU_ANO_CENSO, CO_ENTIDADE)
LEFT JOIN origem.Turma USING (NU_ANO_CENSO, CO_ENTIDADE);

CREATE UNIQUE INDEX censo_chave ON censo_escolar (NU_ANO_CENSO, CO_ENTIDADE);
