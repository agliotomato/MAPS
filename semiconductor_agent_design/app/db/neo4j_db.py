from neo4j import GraphDatabase

from app.config import settings


def _candidate_uris(uri: str) -> list[str]:
    candidates = [uri]
    if uri.startswith("neo4j+s://"):
        candidates.append(uri.replace("neo4j+s://", "bolt+s://", 1))
        candidates.append(uri.replace("neo4j+s://", "bolt+ssc://", 1))
    elif uri.startswith("bolt+s://"):
        candidates.append(uri.replace("bolt+s://", "bolt+ssc://", 1))
    return candidates


def get_neo4j_driver():
    last_err = None
    for candidate in _candidate_uris(settings.neo4j_uri):
        try:
            driver = GraphDatabase.driver(
                candidate,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            driver.verify_connectivity()
            return driver
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Neo4j connection failed: {last_err}")


def ensure_constraints() -> None:
    driver = get_neo4j_driver()
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            "CREATE CONSTRAINT company_code IF NOT EXISTS FOR (c:Company) REQUIRE c.company_code IS UNIQUE"
        )
        session.run(
            "CREATE INDEX tech_event_time IF NOT EXISTS FOR (e:TechEvent) ON (e.published_at)"
        )
        session.run(
            "CREATE INDEX metric_name IF NOT EXISTS FOR (m:Metric) ON (m.name)"
        )
    driver.close()
