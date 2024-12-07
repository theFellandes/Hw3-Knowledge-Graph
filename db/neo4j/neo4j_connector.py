import csv
import io
import re
from dataclasses import dataclass, field

from neo4j import GraphDatabase

# Load Neo4j credentials from environment variables
import os

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your_password"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")


@dataclass
class Neo4jEngine:
    uri: str
    user: str
    password: str
    driver: GraphDatabase.driver = field(init=False)

    def __post_init__(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def __enter__(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.close()

    def close(self):
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

    def insert_into_neo4j(self, entities, relationships):
        session = self.driver.session()
        for entity in entities:
            # Insert entity nodes into Neo4j
            session.run("""
            MERGE (e:Entity {name: $name})
            """, name=entity)

        for relationship in relationships:
            # Insert relationship into Neo4j
            session.run("""
            MATCH (a:Entity {name: $entity1}), (b:Entity {name: $entity2})
            MERGE (a)-[:`$relationship_type`]->(b)
            """, entity1=relationship['entity1'], entity2=relationship['entity2'],
                        relationship_type=relationship['relationship_type'])
        session.close()

    def create_node_updated(self, name):
        query = """
        MERGE (n:Entity {name: $name})
        RETURN n
        """
        with self.driver.session() as session:
            session.run(query, {"name": name})

    def create_relationship_updated(self, source, target, relationship_type):
        query = """
        MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
        MERGE (a)-[r:`RELATIONSHIP` {type: $relationship_type}]->(b)
        RETURN r
        """
        with self.driver.session() as session:
            session.run(query, {"source": source, "target": target, "relationship_type": relationship_type})

    def store_in_neo4j(self, relationships: list):
        with self.driver.session() as session:
            for rel in relationships:
                try:
                    # Ensure the keys are correct before proceeding
                    source = rel.get('source')
                    relationship = rel.get('relationship')
                    target = rel.get('target')

                    # Debugging: Log the relationship being processed
                    print(f"Processing relationship: {rel}")

                    if source and relationship and target:
                        # Create or update source node
                        session.run("MERGE (a:Entity {name: $source})", {"source": source})

                        # Create or update target node
                        session.run("MERGE (b:Entity {name: $target})", {"target": target})

                        # Create or update relationship between nodes
                        session.run("""
                        MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                        MERGE (a)-[r:RELATED {type: $relationship}]->(b)
                        RETURN r
                        """, {"source": source, "target": target, "relationship": relationship})
                    else:
                        print(f"Skipping invalid relationship: {rel}")
                except Exception as e:
                    print(f"Error storing relationship {rel}: {e}")

    def store_named_relationships_from_string(self, csv_content: str):
        """
        Stores named relationships in Neo4j from a CSV string.

        :param csv_content: A string containing the CSV data with headers `source, relationship, target`.
        """
        with self.driver.session() as session:
            try:
                # Use StringIO to simulate a file object from the string
                csv_file = io.StringIO(csv_content)
                reader = csv.reader(csv_file)
                headers = next(reader, None)  # Skip the header row

                # Ensure the CSV has the correct format
                if headers != ["source", "relationship", "target"]:
                    raise ValueError("CSV string must have 'source', 'relationship', and 'target' as headers.")

                for row in reader:
                    if len(row) != 3:  # Ensure each row has exactly 3 elements
                        print(f"Skipping invalid row: {row}")
                        continue

                    # Extract values from the row
                    source, relationship, target = row

                    # Debugging: Log the relationship being processed
                    print(f"Processing relationship: source={source}, relationship={relationship}, target={target}")

                    if source and relationship and target:
                        # Create or update source node
                        session.run("MERGE (a:Entity {name: $source})", {"source": source})

                        # Create or update target node
                        session.run("MERGE (b:Entity {name: $target})", {"target": target})

                        # Create or update named relationship between nodes
                        query = f"""
                        MATCH (a:Entity {{name: $source}}), (b:Entity {{name: $target}})
                        MERGE (a)-[r:{relationship}]->(b)
                        RETURN r
                        """
                        session.run(query, {"source": source, "target": target})
                    else:
                        print(f"Skipping invalid relationship: {row}")
            except Exception as e:
                print(f"Error processing CSV content: {e}")

    def store_named_relationships_from_file(self, csv_file_path: str):
        """
        Stores named relationships in Neo4j from a CSV file.

        :param csv_file_path: Path to the CSV file containing relationships with headers `source, relationship, target`.
        """
        with self.driver.session() as session:
            try:
                with open(csv_file_path, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    headers = next(reader, None)  # Skip the header row

                    # Ensure the CSV has the correct format
                    if headers != ["source", "relationship", "target"]:
                        raise ValueError("CSV file must have 'source', 'relationship', and 'target' as headers.")

                    for row in reader:
                        if len(row) != 3:  # Ensure each row has exactly 3 elements
                            print(f"Skipping invalid row: {row}")
                            continue

                        # Extract values from the row
                        source, relationship, target = row

                        # Sanitize the relationship type (replace invalid characters)
                        sanitized_relationship = re.sub(r"[^a-zA-Z0-9_]", "_", relationship).upper()

                        # Debugging: Log the relationship being processed
                        print(
                            f"Processing relationship: source={source}, relationship={sanitized_relationship}, target={target}")

                        if source and sanitized_relationship and target:
                            # Create or update source node
                            session.run("MERGE (a:Entity {name: $source})", {"source": source})

                            # Create or update target node
                            session.run("MERGE (b:Entity {name: $target})", {"target": target})

                            # Create or update named relationship between nodes
                            query = f"""
                            MATCH (a:Entity {{name: $source}}), (b:Entity {{name: $target}})
                            MERGE (a)-[r:{sanitized_relationship}]->(b)
                            RETURN r
                            """
                            session.run(query, {"source": source, "target": target})
                        else:
                            print(f"Skipping invalid relationship: {row}")
            except Exception as e:
                print(f"Error processing CSV file {csv_file_path}: {e}")

    def store_in_neo4j_csv(self, csv_file_path: str):
        """
        Stores relationships in Neo4j from a CSV file.

        :param csv_file_path: Path to the CSV file containing relationships
        """
        with self.driver.session() as session:
            try:
                with open(csv_file_path, mode='r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)  # Read CSV with headers as dictionary
                    for rel in reader:
                        # Ensure the keys are correct before proceeding
                        source = rel.get('source')
                        relationship = rel.get('relationship')
                        target = rel.get('target')

                        # Debugging: Log the relationship being processed
                        print(f"Processing relationship: {rel}")

                        if source and relationship and target:
                            # Create or update source node
                            session.run("MERGE (a:Entity {name: $source})", {"source": source})

                            # Create or update target node
                            session.run("MERGE (b:Entity {name: $target})", {"target": target})

                            # Create or update relationship between nodes
                            session.run("""
                            MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                            MERGE (a)-[r:RELATED {type: $relationship}]->(b)
                            RETURN r
                            """, {"source": source, "target": target, "relationship": relationship})
                        else:
                            print(f"Skipping invalid relationship: {rel}")
            except Exception as e:
                print(f"Error reading or processing CSV file {csv_file_path}: {e}")
