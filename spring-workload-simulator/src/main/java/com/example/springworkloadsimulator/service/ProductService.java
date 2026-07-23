package com.example.springworkloadsimulator.service;

import com.example.springworkloadsimulator.model.Product;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

@Service
public class ProductService {

    private final Map<Long, Product> store = new HashMap<>();
    private final AtomicLong idSeq = new AtomicLong(1);

    public ProductService() {
        save(new Product(null, "Laptop", 999.99));
        save(new Product(null, "Mouse", 29.99));
        save(new Product(null, "Keyboard", 49.99));
    }

    public List<Product> findAll() {
        return new ArrayList<>(store.values());
    }

    public Optional<Product> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    public Product save(Product product) {
        if (product.getId() == null) {
            product.setId(idSeq.getAndIncrement());
        }
        store.put(product.getId(), product);
        return product;
    }

    public boolean delete(Long id) {
        return store.remove(id) != null;
    }
}
