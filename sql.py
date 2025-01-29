from sqlalchemy import create_engine, Column, Integer, String, DateTime, select
from sqlalchemy.orm import sessionmaker, declarative_base  # Correct import here
from datetime import datetime

# Step 1: Define the database URI and the engine
DATABASE_URI = "sqlite:///C:/Users/pkole/Downloads/zoho.db"  # Replace with your actual path

# Step 2: Create an engine and session
engine = create_engine(DATABASE_URI, echo=True)  # echo=True to see the SQL statements
Session = sessionmaker(bind=engine)
session = Session()

# Step 3: Define the ORM model for your table
Base = declarative_base()

class CustomerOrder(Base):
    __tablename__ = 'customer_orders'  # Your actual table name

    order_id = Column(Integer, primary_key=True)
    customer_id = Column(Integer)
    pizza_id = Column(Integer)
    exclusions = Column(String(4))
    extras = Column(String(4))
    order_time = Column(DateTime)

    def __repr__(self):
        return f"<CustomerOrder(order_id={self.order_id}, customer_id={self.customer_id}, pizza_id={self.pizza_id}, order_time={self.order_time})>"
        # return f"<co(oi={self.order_id}, ci={self.customer_id}, pi={self.pizza_id}, ot={self.order_time})>"



# Step 4: Test the date filtering
try:
    # Define the date to filter by
    filter_date = datetime(2020, 1, 8, 0, 0, 0)

    # Create a query to get orders placed after the filter_date
    query = select(CustomerOrder).where(CustomerOrder.order_time > filter_date)

    # Execute the query and fetch the results
    result = session.execute(query).scalars().all()

    # Step 5: Print the results
    if result:
        for order in result:
            print(order)
    else:
        print("No orders found after the specified date.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Step 6: Close the session
    session.close()
