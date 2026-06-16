-- Create a table with all FIFA World Cup matches after the year 2000
CREATE TABLE world_cups AS FROM read_csv('data/results.csv',
    header=True,
    delim=',',
    nullstr='NA'
) WHERE tournament = 'FIFA World Cup' AND year(date) > 2000;

-- Query to get the match details along with the stage and round of the tournament
WITH wc AS (
    SELECT *,
           row_number() OVER (ORDER BY date) AS match_id,
           date <= DATE '2002-06-14' AS is_group
    FROM world_cups
    WHERE tournament = 'FIFA World Cup'
      AND date BETWEEN DATE '2002-05-31' AND DATE '2002-06-30'
),
appearances AS (
    SELECT match_id, date, home_team AS team FROM wc
    UNION ALL
    SELECT match_id, date, away_team AS team FROM wc
),
team_round AS (
    SELECT match_id,
           row_number() OVER (PARTITION BY team ORDER BY date, match_id) AS team_game_no
    FROM appearances
)
SELECT wc.date, wc.home_team, wc.away_team, wc.home_score, wc.away_score,
       CASE WHEN wc.is_group THEN 'Group' ELSE 'Knockout' END AS stage,
       CASE WHEN wc.is_group THEN tr.team_game_no ELSE (tr.team_game_no - 3) END AS round
FROM wc
JOIN team_round tr USING (match_id);