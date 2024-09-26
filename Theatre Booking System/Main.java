import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;
import java.util.Random;

//seat class that consists of information of individual seat
class Seat{
    private int seatNumber;
    private boolean isBooked;
    private boolean isSelected;
    private ReentrantLock lock;
    private int isBookedBy;

    //constructor
    public Seat(int seatNumber) {
        this.seatNumber = seatNumber;
        this.isBooked = false;
        this.isSelected = false;
        this.lock = new ReentrantLock();
    }

    //book seat function, returns success or not success
    public boolean bookSeat() {
        Random random = new Random();
        lock.lock();
        isSelected = true;
        try {
            //check if it has been booked
            if (!isBooked) {
                // before reservation make sure to have a delay of 500-1000ms
                try {
                    Thread.sleep(random.nextInt(501) + 500);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                //This seat has been booked
                isBooked = true;
                isSelected = false;
                return true;
            }
            return false;
        } finally {
            lock.unlock();
        }
    }

    //setter and getter for this class
    public boolean isBooked() {
        return isBooked;
    }

    public boolean isSelected() {
        return isSelected;
    }

    public int getSeatNumber() {
        return seatNumber;
    }

    public void SetIsBookedBy(int customerId){
        this.isBookedBy = customerId;
    }

    public int getIsBookedBy(){
        return isBookedBy;
    }
}

//Theater class
class Theatre{
    private int theatreId;
    private boolean isFullyBooked;
    private Seat[] seats;
    private List<Seat> availableSeats;
    private List<Booking> bookings;
    private ReentrantLock lock;

    //constructor
    public Theatre (int theatreId, int numberOfSeats){
        this.theatreId = theatreId;
        this.seats = new Seat[numberOfSeats];
        this.availableSeats = new ArrayList<>();
        this.isFullyBooked = false;
        this.bookings = new ArrayList<>();
        this.lock = new ReentrantLock();
        for (int i = 0; i < numberOfSeats; i++){
            seats[i] = new Seat(i+1);
            this.availableSeats.add(seats[i]);
        }
    }

    //according to customer's request book seats
    public List<Integer> bookSeats (int customerId, int noOfSeats){
        List<Integer> bookedSeats = new ArrayList<>();
        // get the available seats for booking
        List<Seat> availableSeats = getAvailableSeats();
        try {
            // select the number of seats from the available seats and book seats
            if (availableSeats.size() >= noOfSeats) {
                for (int i = 0; i < noOfSeats; i++) {
                    Seat seat = availableSeats.get(i);
                    if (seat.bookSeat()) {
                        bookedSeats.add(seat.getSeatNumber());
                    }
                }
            }
            // if seats are successfully added, add a new booking
            if (bookedSeats.size() > 0){
                Booking booking = new Booking(customerId, theatreId, bookedSeats);
                bookings.add(booking);
            }
            return bookedSeats;
        } finally {

        }
    }

    //setter and getters
    public int getTheatreId(){
        return theatreId;
    }

    public List<Booking> getBookings(){
        return bookings;
    }

    public Seat[] getSeats(){
        return seats;
    }

    //get all available seats, if no more available seats, set FullyBooked to true
    public synchronized List<Seat> getAvailableSeats(){
        List<Seat> availableSeats = new ArrayList<>();
        for (Seat seat : seats) {
            if (!seat.isBooked() || !seat.isSelected()) {
                availableSeats.add(seat);
            }
        }
        if (availableSeats.size() == 0) {
            setFullyBooked(true);
        }
        return availableSeats;
    }

    public synchronized boolean isFullyBooked() {
        return isFullyBooked;
    }

    private synchronized void setFullyBooked(boolean fullyBooked) {
        isFullyBooked = fullyBooked;
    }
}

class Customer extends Thread{
    private int customerId;
    private List<Theatre> theatres;
    public Customer(int customerId, List<Theatre> theatres){
        this.customerId = customerId;
        this.theatres = theatres;
    }

    @Override
    public void run(){
        Random random = new Random();
        List<Integer> bookedSeats = new ArrayList<>();
        //customer selects theater room and seats to book
        int theaterRoom = random.nextInt(theatres.size());
        int noSeatsToBook = random.nextInt(3) + 1;
        Theatre theatre = theatres.get(theaterRoom);
        //if customer does not manage to book seats and the theater is not fully booked, attempt booking again
        while (bookedSeats.size() == 0 && !theatre.isFullyBooked()) {
            bookedSeats = theatre.bookSeats(customerId, noSeatsToBook);
            if (theatre.isFullyBooked()){
                System.out.println(customerId + ": Theater " + + theatre.getTheatreId() + " is fully booked");
                break;
            }
            if (bookedSeats.size() != 0) {
                System.out.println("Customer " + customerId + " successfully booked " + bookedSeats.size() + " seat(s) in Theatre " + theatre.getTheatreId() + " , initially decided to book " + noSeatsToBook + " seat(s)");
                System.out.println("Seats booked: ");
                for (int i =0; i<bookedSeats.size(); i++){
                    System.out.println(bookedSeats.get(i));
                }
            }
        }
    }
}

// records booking history
class Booking{
    private int customerId;
    private int theatreId;
    private List<Integer> seatsBooked;

    public Booking(int customerId, int theatreId, List<Integer> seats){
        this.customerId = customerId;
        this.theatreId = theatreId;
        this.seatsBooked = seats;
    }

    public String toString() {
        return "Booking [customerId=" + customerId + ", theatreId=" + theatreId + ", seatsBooked=" + seatsBooked + "]";
    }
}

//main class
public class Main {
    public static void main(String[] args) {
        int noOfTheatres = 3;
        int seatsPerTheatre = 20;
        int noOfCustomers = 100;

        //create theatre classes
        List<Theatre> theatres = new ArrayList<>();
        for (int i = 0; i < noOfTheatres; i++) {
            theatres.add(new Theatre(i + 1, seatsPerTheatre));
        }

        //create customer threads
        List<Customer> customers = new ArrayList<>();
        for (int i = 0; i < noOfCustomers; i++) {
            customers.add(new Customer((i + 1), theatres));
        }

        //start threads running
        for (Customer customer : customers) {
            customer.start();
        }

        //wait for threads to finish running
        for (Customer customer : customers) {
            try {
                customer.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        //print out theatres and the total bookings
        for (int i = 0; i < noOfTheatres; i++) {
            Theatre theatre = theatres.get(i);
            System.out.println("Theater: " + theatre.getTheatreId());
            System.out.println(theatre.getBookings());
        }
        System.out.println("All customers have finished booking.");
    }
}