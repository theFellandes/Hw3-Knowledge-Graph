from dataclasses import dataclass, field

from neo4j import GraphDatabase

# Load Neo4j credentials from environment variables
import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")


@dataclass
class Neo4jEngine:
    uri: str
    user: str
    password: str
    driver: GraphDatabase.driver = field(init=False)

    def __enter__(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.close()

    def run(self, query, parameters=None):
        with self.driver.session() as session:
            return session.run(query, parameters or {})

    def create_node(self, node_type, properties):
        query = f"""
        MERGE (n:{node_type} {{name: $name}})
        SET n += $props
        RETURN n
        """
        self.run(query, {"name": properties["name"], "props": properties})

    def create_relationship(self, source, target, relationship):
        query = f"""
        MATCH (a {{name: $source}}), (b {{name: $target}})
        MERGE (a)-[r:{relationship}]->(b)
        RETURN r
        """
        self.run(query, {"source": source, "target": target})

